import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import numpy as np

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import math
from PIL import Image
from transformers import set_seed
import seaborn as sns
import matplotlib.pyplot as plt

from eval_scripts.eval_utils.head_attribution import set_zero_ablation_greedy_search
set_zero_ablation_greedy_search()

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def json_custom_serializer(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError("Type %s not serializable" % type(obj))

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = []
    sampled_img_ids = []
    with open(os.path.expanduser(args.answers_file), "r") as f:
        caps = json.load(f)["sentences"]
        for cap in caps:
            if cap["metrics"]["CHAIRs"] == 1:
                image_id = "{:012d}".format(cap["image_id"])
                image_file = f"COCO_train2014_{image_id}.jpg"
                question = {
                    "question_id": cap["image_id"],
                    "image": image_file,
                    "image_path": os.path.join(args.image_folder, image_file),
                    "text": "Please describe this image in detail.",
                    "caption": cap["caption"],
                    "mscoco_hallucinated_words": [i[0] for i in cap["mscoco_hallucinated_words"]],
                    "mscoco_non_hallucinated_words": [i[0] for i in cap["mscoco_non_hallucinated_words"]],
                }
                questions.append(question)
                sampled_img_ids.append(image_id)
    print(len(questions))

    questions = questions[args.start_idx:args.end_idx]
    sampled_img_ids = sampled_img_ids[args.start_idx:args.end_idx]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)
    os.makedirs(args.output_path, exist_ok=True)

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        print(line)
        question_id = line["question_id"]
        image_file = line["image"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        image_tensor = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)

        hallucinated_ids = [tokenizer(word)['input_ids'] for word in line['mscoco_hallucinated_words']]
        hallucinated_tokens = [tokenizer.decode(hallucinated_id[1]) for hallucinated_id in hallucinated_ids]
        non_hallucinated_ids = [tokenizer(word)['input_ids'] for word in line['mscoco_non_hallucinated_words']]
        non_hallucinated_tokens = [tokenizer.decode(non_hallucinated_id[1]) for non_hallucinated_id in non_hallucinated_ids]

        with torch.inference_mode():
            _, hallucination_influences, non_hallucination_influences = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                output_attentions=True,
                return_dict_in_generate=True, 
                hallucinated_tokens=hallucinated_tokens, 
                non_hallucinated_tokens=non_hallucinated_tokens,
                influence_score=args.influence_score)
            
        torch.save(hallucination_influences, os.path.join(args.output_path, f'hallucination_influences_{question_id}.pth'))
        torch.save(non_hallucination_influences, os.path.join(args.output_path, f'non_hallucination_influences_{question_id}.pth'))
        torch.cuda.empty_cache()

        influences = []
        for _, v in hallucination_influences.items():
            influence = torch.zeros(args.layer_num, args.head_num)
            for layer_idx in range(args.layer_num):
                for head_idx in range(args.head_num):
                    influence[layer_idx][head_idx] = v[layer_idx][head_idx]['influence']
            influences.append(influence)
        plt.figure(figsize=(8,8))
        sns.heatmap(torch.mean(torch.stack(influences), 0).cpu().numpy(),cmap="coolwarm", center=0)
        plt.savefig(f'{args.output_path}/hallucination_influences_{question_id}.png')
        
        influences = []
        for _, v in non_hallucination_influences.items():
            influence = torch.zeros(args.layer_num, args.head_num)
            for layer_idx in range(args.layer_num):
                for head_idx in range(args.head_num):
                    influence[layer_idx][head_idx] = v[layer_idx][head_idx]['influence']
            influences.append(influence)
        plt.figure(figsize=(8,8))
        sns.heatmap(torch.mean(torch.stack(influences), 0).cpu().numpy(),cmap="coolwarm", center=0)
        plt.savefig(f'{args.output_path}/non_hallucination_influences_{question_id}.png')
        
    
def get_constrative_influence(args):

    files = os.listdir(args.output_path)
    hallucination_samples = []
    non_hallucination_samples = []
    for file in files:
        if file.endswith('pth'):
            if file.startswith('hal'):
                hallucination_sample = torch.load(os.path.join(args.output_path, file))
                influences = []
                for _, v in hallucination_sample.items():
                    influence = torch.zeros(args.layer_num, args.head_num)
                    for layer_idx in range(args.layer_num):
                        for head_idx in range(args.head_num):
                            influence[layer_idx][head_idx] = v[layer_idx][head_idx]['influence']
                    influences.append(influence)
                hallucination_samples += influences
            else:
                non_hallucination_sample = torch.load(os.path.join(args.output_path, file))
                influences = []
                for _, v in non_hallucination_sample.items():
                    influence = torch.zeros(args.layer_num, args.head_num)
                    for layer_idx in range(args.layer_num):
                        for head_idx in range(args.head_num):
                            influence[layer_idx][head_idx] = v[layer_idx][head_idx]['influence']
                    influences.append(influence)
                non_hallucination_samples += influences

    hallucinated_scores = torch.stack(hallucination_samples)
    non_hallucinated_scores = torch.stack(non_hallucination_samples)

    plt.figure(figsize=(8,8))
    difference = torch.mean(hallucinated_scores,0).float() - torch.mean(non_hallucinated_scores,0).float() 
    ax = sns.heatmap(difference.cpu().numpy(),cmap="coolwarm", center=0)
    ax.set_xlabel('Head Index', fontsize=12)
    ax.set_ylabel('Layer Index', fontsize=12)
    print(f'{args.output_path}/constrastive_influences.png')
    plt.savefig(f'{args.output_path}/constrastive_influences.png')

    results = {}
    _, flat_indices = torch.topk(difference.flatten(), args.topk, largest=True)
    indices = [[flat_indice.numpy() // 32, flat_indice.numpy() % 32] for flat_indice in flat_indices]
    results.update({'hal_heads': indices})
    _, flat_indices = torch.topk(difference.flatten(), args.topk, largest=False)
    indices =  [[flat_indice.numpy() // 32, flat_indice.numpy() % 32] for flat_indice in flat_indices]
    results.update({'non_hal_heads': indices})
    print(results)

    print(f'{args.output_path}/attribution_result.json')
    with open(f'{args.output_path}/attribution_result.json', 'w') as file:
        json.dump(results, file, default=json_custom_serializer) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--annotation-dir", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=10000)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--influence_score", type=str, default='prob_diff')
    parser.add_argument("--layer_num", type=int, default=32)
    parser.add_argument("--head_num", type=int, default=32)
    args = parser.parse_args()
    set_seed(args.seed)
    
    # get hallucination and non-hallucination influences for each question
    eval_model(args)
    # average the influences over all questions
    get_constrative_influence(args)
