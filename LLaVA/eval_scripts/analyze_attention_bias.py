import argparse
import torch
import os
import json
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria
from torch.utils.data import Dataset, DataLoader
from transformers import set_seed

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


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
        caption = line["caption"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() + ' ' + caption

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


def calculate_mean(hal_img_attns_all):
    values = []
    for hal_img_attns in hal_img_attns_all:
        for key, value in hal_img_attns.items():
            values.append(value)
    mean_value = torch.mean(torch.tensor(values))
    return mean_value

def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    questions = []
    with open(os.path.expanduser(args.answers_file), "r") as f:
        caps = json.load(f)["sentences"]
        for cap in caps:
            if cap["metrics"]["CHAIRs"] == 1:
                image_id = "{:012d}".format(cap["image_id"])
                image_file = f"COCO_train2014_{image_id}.jpg"
                question = {
                    "question_id": cap["image_id"],
                    "image": image_file,
                    "text": "Please describe this image in detail.",
                    "caption": cap["caption"],
                    "mscoco_generated_words": cap["mscoco_generated_words"],
                }
                questions.append(question)
                

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    count = 0
    with open(args.attention_head_path, 'r') as file:
        data_loaded = json.load(file)
    hal_attention_heads = data_loaded['hal_heads'][:args.top_k]
    non_hal_attention_heads = data_loaded['non_hal_heads'][:args.top_k]

    print(hal_attention_heads)
    print(non_hal_attention_heads)
    attention_statics = []
    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        count += 1
        question_id = line["question_id"]
        image_file = line["image"]
        mscoco_generated_words = line['mscoco_generated_words'] 
        mscoco_generated_words_first_token = [tokenizer.decode([tokenizer(generated_word)['input_ids'][1]]) for generated_word in mscoco_generated_words]

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        image_tensor = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True
                )

        output_attentions = outputs['attentions'][0]
        image_start_idx = torch.nonzero(input_ids[0]==IMAGE_TOKEN_INDEX)[0][0]
        ids = input_ids[0, image_start_idx+1:].cpu().numpy().tolist()
        text_tokens = [tokenizer.decode([id]) for id in ids]

        image_length = 576
        for idx, text_token in enumerate(text_tokens):
            if text_token in mscoco_generated_words_first_token: # if the text token is an object in the generated caption
                hal_img_attn = 0.0
                hal_txt_attn = 0.0
                for layer_idx, head_idx in hal_attention_heads:
                    attention_scores = output_attentions[layer_idx][0][0, head_idx, :, :].clone()
                    img_attention_score = attention_scores[idx + image_start_idx + image_length - 1, image_start_idx:image_start_idx + image_length]
                    txt_attention_score = attention_scores[idx + image_start_idx + image_length - 1, image_start_idx + image_length:image_start_idx + image_length + idx]
                    hal_img_attn += torch.sum(img_attention_score) / len(hal_attention_heads)
                    hal_txt_attn += torch.sum(txt_attention_score) / len(hal_attention_heads)

                nonhal_img_attn = 0.0
                nonhal_txt_attn = 0.0
                for layer_idx, head_idx in non_hal_attention_heads:
                    attention_scores = output_attentions[layer_idx][0][0, head_idx, :, :].clone()
                    img_attention_score = attention_scores[idx + image_start_idx + image_length - 1, image_start_idx:image_start_idx + image_length]
                    txt_attention_score = attention_scores[idx + image_start_idx + image_length - 1, image_start_idx + image_length:image_start_idx + image_length + idx]
                    nonhal_img_attn += torch.sum(img_attention_score) / len(non_hal_attention_heads)
                    nonhal_txt_attn += torch.sum(txt_attention_score) / len(non_hal_attention_heads)
                
                attention_statics.append([hal_img_attn.item(), hal_txt_attn.item(), nonhal_img_attn.item(), nonhal_txt_attn.item()])
        
        if count == 100:
            break
    
    os.makedirs(args.output_path, exist_ok=True)
    torch.save(attention_statics, os.path.join(args.output_path, 'attention_statics.pth'))

def plot_result(args):
    
    attention_statics = torch.load(os.path.join(args.output_path, 'attention_statics.pth'))
    hal_img_attn = torch.mean(torch.tensor([item[0] for item in attention_statics]))
    hal_txt_attn = torch.mean(torch.tensor([item[1] for item in attention_statics]))
    nonhal_img_attn = torch.mean(torch.tensor([item[2] for item in attention_statics]))
    nonhal_txt_attn = torch.mean(torch.tensor([item[3] for item in attention_statics]))

    print(hal_img_attn, hal_txt_attn, nonhal_img_attn, nonhal_txt_attn)
    
    labels = ['Hallucination Heads', 'Non-Hallucination Heads']
    text_attention = [hal_txt_attn, nonhal_txt_attn]
    image_attention = [hal_img_attn, nonhal_img_attn]
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot bars
    ax.bar(x - width/2, text_attention, width, label='Text Attention', color='skyblue')
    ax.bar(x + width/2, image_attention, width, label='Image Attention', color='lightslategray')

    # Add labels, title, and gridlines
    ax.set_ylabel('Attention Weights', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=16)
    ax.set_ylim(0, 1)  # Ensure y-axis starts at 0
    ax.grid(axis='y', linestyle='--', alpha=0.7)  # Add horizontal gridlines
    ax.legend(loc='upper left', fontsize=16)  # Adjust legend location

    plt.tight_layout()
    print(os.path.join(args.output_path, 'bar.png'))
    plt.savefig(os.path.join(args.output_path, 'bar.png'))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--annotation-dir", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="answers.jsonl")
    parser.add_argument("--attention_head_path", type=str, default="")
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
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()
    set_seed(args.seed)

    eval_model(args)
    plot_result(args)
