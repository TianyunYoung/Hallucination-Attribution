import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

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
import matplotlib.colors as Colormap
from matplotlib.colors import LogNorm
from nltk.corpus import wordnet

import numpy as np
from transformers import LlamaForCausalLM
import copy


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

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

        return input_ids, image_tensor, image.size, prompt

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, prompts = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes, prompts


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader

def js_divergence(P, Q, dim=-1, epsilon=1e-8):
    # Normalize the tensors along the specified dimension
    P_normalized = P / P.sum(dim=dim, keepdim=True)
    Q_normalized = Q / Q.sum(dim=dim, keepdim=True)
    
    # Compute the midpoint distribution
    M = 0.5 * (P_normalized + Q_normalized)
    
    # Add epsilon to avoid log(0) or division by zero
    P_normalized = P_normalized + epsilon
    Q_normalized = Q_normalized + epsilon
    M = M + epsilon
    
    # Compute KL divergences
    KL_P_M = (P_normalized * (P_normalized.log() - M.log())).sum(dim=dim)
    KL_Q_M = (Q_normalized * (Q_normalized.log() - M.log())).sum(dim=dim)
    
    # Compute JS divergence
    JS_divergence = 0.5 * (KL_P_M + KL_Q_M)
    
    return JS_divergence

def attention_map_vis(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    ## load fine-tuned model
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name, device_map='cuda:0', device='cuda:0')

    ## copy fine-tuned model as pretrained model
    _, pretrained_model, _, _ = load_pretrained_model(model_path, args.model_base, model_name, device_map='cuda:1', device='cuda:1')
    # load mm projector weights
    mm_projector_weights = torch.load('../checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin', map_location='cuda:1')
    mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
    pretrained_model.load_state_dict(mm_projector_weights, strict=False)
    # load llm model weights
    llm_model_weights = LlamaForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5", torch_dtype=torch.float16, device_map="cuda:1")
    llm_model_weights = {k: v.to(torch.float16) for k, v in llm_model_weights.state_dict().items()}
    pretrained_model.load_state_dict(llm_model_weights, strict=False)

    del llm_model_weights, mm_projector_weights

    questions = []
    sampled_img_ids = []
    with open(os.path.expanduser(args.question_file), "r") as f:
        caps = json.load(f)["sentences"]
        for cap in caps:
            image_id = "{:012d}".format(cap["image_id"])
            image_file = f"COCO_train2014_{image_id}.jpg"
            question = {
                "question_id": cap["image_id"],
                "image": image_file,
                "image_path": os.path.join(args.image_folder, image_file),
                "text": "Please describe this image in detail.",
                "caption": cap["caption"],
            }
            questions.append(question)
            sampled_img_ids.append(image_id)
    print(len(questions))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)
    count = 0
    os.makedirs(args.output_path, exist_ok=True)

    with open(args.attention_head_path, 'r') as file:
        data_loaded = json.load(file)
    hal_attention_heads = data_loaded['hal_heads'][:args.top_k]
    non_hal_attention_heads = data_loaded['non_hal_heads'][:args.top_k]
    

    hal_heads_js_div_all_samples = []
    non_hal_heads_js_div_all_samples = []
    for (input_ids, image_tensor, image_sizes, prompts), line in tqdm(zip(data_loader, questions), total=len(questions)):
        # print(line)
        count += 1
        question_id = line["question_id"]
        image_file = line["image"]

        # Fine-tuned LVLM generate
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        image_tensor = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)
        with torch.inference_mode():
            outputs1 = model.generate(
                                input_ids,
                                images=image_tensor,
                                image_sizes=image_sizes,
                                max_new_tokens=args.max_new_tokens,
                                use_cache=True,
                                output_attentions=True,
                                output_scores=True,
                                output_hidden_states=True,
                                return_dict_in_generate=True)
    
        # Pretrained LVLM generate
        input_ids = input_ids.to(pretrained_model.device)
        image_tensor = image_tensor.to(pretrained_model.device)
        with torch.inference_mode():
            outputs2 = pretrained_model.generate(
                                input_ids,
                                images=image_tensor,
                                image_sizes=image_sizes,
                                max_new_tokens=args.max_new_tokens,
                                use_cache=True,
                                output_attentions=True,
                                output_scores=True,
                                output_hidden_states=True,
                                return_dict_in_generate=True)
            
        output_attentions1 = outputs1['attentions'][0]
        output_attentions2 = outputs2['attentions'][0]

        hal_heads_js_div = []
        for idx, attn_head in enumerate(hal_attention_heads):
            layer_idx, head_idx = attn_head
            attention_map1 = output_attentions1[layer_idx][0][head_idx].cpu().float() # fine-tuned LVLM
            attention_map2 = output_attentions2[layer_idx][0][head_idx].cpu().float() # pretrained LVLM

            js = []
            for _, (attn1, attn2) in enumerate(zip(attention_map1, attention_map2)):
                js.append(js_divergence(attn1, attn2))
            hal_heads_js_div.append(torch.mean(torch.tensor(js)))

        non_hal_heads_js_div = []
        for idx, attn_head in enumerate(non_hal_attention_heads):
            layer_idx, head_idx = attn_head
            attention_map1 = output_attentions1[layer_idx][0][head_idx].cpu().float() # fine-tuned LVLM
            attention_map2 = output_attentions2[layer_idx][0][head_idx].cpu().float() # pretrained LVLM

            js = []
            for _, (attn1, attn2) in enumerate(zip(attention_map1, attention_map2)):
                js.append(js_divergence(attn1, attn2))
            non_hal_heads_js_div.append(torch.mean(torch.tensor(js)))

        # print(hal_heads_js_div) 
        # print(non_hal_heads_js_div) 
        hal_heads_js_div_all_samples.append(torch.mean(torch.tensor(hal_heads_js_div)))
        non_hal_heads_js_div_all_samples.append(torch.mean(torch.tensor(non_hal_heads_js_div)))

        torch.cuda.empty_cache()
        
        if count >= 50:
            break
    
    torch.save(hal_heads_js_div_all_samples, os.path.join(args.output_path, 'hal_heads_js_div.pth'))
    torch.save(non_hal_heads_js_div_all_samples, os.path.join(args.output_path, 'non_hal_heads_js_div.pth'))
    print('hal_heads_js_div:', torch.mean(torch.tensor(hal_heads_js_div_all_samples)))
    print('non_hal_heads_js_div:', torch.mean(torch.tensor(non_hal_heads_js_div_all_samples)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--annotation-dir", type=str, default="")
    parser.add_argument("--question-file", type=str, default="question.jsonl")
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
    parser.add_argument("--start_idx", type=int, default=20)
    parser.add_argument("--end_idx", type=int, default=10000)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--attention_head_path", type=str, default=None)

    args = parser.parse_args()
    set_seed(args.seed)
    attention_map_vis(args)
