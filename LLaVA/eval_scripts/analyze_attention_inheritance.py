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

from transformers import set_seed
from transformers import LlamaForCausalLM

import math
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as Colormap
from matplotlib.colors import LogNorm
from nltk.corpus import wordnet

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def is_noun(word):
    synsets = wordnet.synsets(word, pos=wordnet.NOUN)
    for synset in synsets:
        if 'artifact' in synset.lexname() or 'food' in synset.lexname() or 'plant' in synset.lexname():  # 'artifact' is a category for physical objects
            return True
    return False

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

# Function to get token embeddings
def get_token_embeddings(tokenizer, model):
    tokens = tokenizer.get_vocab()
    token_list = list(tokens.keys())
    token_ids = torch.tensor(list(tokens.values())).to('cuda')
    token_embeddings = model.get_model().embed_tokens(token_ids).detach()
    return token_list, token_embeddings


# Function to find the most similar word
def find_most_similar_word(image_embedding, token_list, token_embeddings):
    image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
    token_embeddings = token_embeddings / token_embeddings.norm(dim=-1, keepdim=True)
    similarities = torch.matmul(image_embedding, token_embeddings.T)
    most_similar_index = torch.argmax(similarities, 1) 
    most_similar_word = [token_list[i] for i in most_similar_index]
    
    return most_similar_word

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
        prompt = conv.get_prompt() + caption

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


def visualize_attention(attention_map, text_tokens, output_path="atten_map_1.png", title="Layer 5", stride=20):

    averaged_attention = torch.nn.functional.avg_pool2d(attention_map.unsqueeze(0).unsqueeze(0), stride, stride=stride).squeeze(0).squeeze(0)
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(6, 6))
    log_norm = LogNorm(vmin=0.0007, vmax=averaged_attention.max())
    ax = sns.heatmap(averaged_attention, cmap="coolwarm", norm=log_norm, vmin=1e-3 , vmax=1e-1)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)  # Set the font size for colorbar ticks

    plt.title(title, fontsize=35)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')


def visualize_attention_compare(attention_maps, layer_idx, head_idx, output_path, title, stride=20):
    
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Create a 1x2 grid of subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    vmin = 1e-3  # Minimum value for color scale
    vmax = 1e-1  # Maximum value for color scale

    # LVLM attention map
    averaged_attention = torch.nn.functional.avg_pool2d(attention_maps[0].unsqueeze(0).unsqueeze(0), \
                                                        stride, stride=stride).squeeze(0).squeeze(0)
    log_norm = LogNorm(vmin=0.0007, vmax=averaged_attention.max())
    sns.heatmap(averaged_attention, ax=axs[0],  cmap="coolwarm", norm=log_norm, vmin=vmin, vmax=vmax)
    axs[0].set_title("LLaVA 1.5-7B", fontsize=16)

    # LLM attention map
    averaged_attention = torch.nn.functional.avg_pool2d(attention_maps[1].unsqueeze(0).unsqueeze(0), \
                                                        stride, stride=stride).squeeze(0).squeeze(0)
    sns.heatmap(averaged_attention, ax=axs[1],  cmap="coolwarm", norm=log_norm, vmin=vmin, vmax=vmax)
    axs[1].set_title("Vicuna-7B", fontsize=16)

    plt.tight_layout()
    fig.suptitle(title, fontsize=16, y=1.1)
    plt.savefig(output_path, bbox_inches='tight')

def cosine_similarity(attention_map_1, attention_map_2):
    sim = torch.nn.functional.cosine_similarity(attention_map_1.unsqueeze(0), attention_map_2.unsqueeze(0))
    return sim

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
            # if cap["metrics"]["CHAIRs"] == 1:
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
    
    non_hal_similarities_all_samples = []
    hal_similarities_all_samples = []
    llm_model = LlamaForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5", torch_dtype=torch.float16, device_map="cuda:1")
    for (input_ids, image_tensor, image_sizes, prompts), line in tqdm(zip(data_loader, questions), total=len(questions)):
        print(line)
        count += 1
        question_id = line["question_id"]
        image_file = line["image"]

        # LVLM generate
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
    
        # LLM generate
        llm_input_ids = tokenizer(prompts[0], return_tensors="pt", add_special_tokens=True).input_ids.to(llm_model.device)
        with torch.inference_mode():
            input_embeds = llm_model.get_input_embeddings()(llm_input_ids)
            outputs2 = llm_model.generate(inputs_embeds=input_embeds, 
                                        use_cache=True,
                                        output_attentions=True,
                                        output_scores=True,
                                        output_hidden_states=True,
                                        return_dict_in_generate=True)

        output_attentions1 = outputs1['attentions'][0]
        output_attentions2 = outputs2['attentions'][0]
        
        text_input_ids = tokenizer(prompts[0].split('<image>\n')[1], return_tensors="pt", add_special_tokens=True).input_ids.to(llm_model.device)        
        text_tokens = [tokenizer.decode([id]) for id in text_input_ids[0]]
        print(text_tokens)
        
        stride = 10
        hal_similarities = []
        for idx, attn_head in enumerate(hal_attention_heads):
            layer_idx, head_idx = attn_head
            attention_map1 = output_attentions1[layer_idx][0][0][head_idx].cpu().float()
            attention_map1 = attention_map1[-text_input_ids.shape[1]:, -text_input_ids.shape[1]:] # get the text part of the attention map

            attention_map2 = output_attentions2[layer_idx][0][head_idx].cpu().float()
            attention_map2 = attention_map2[-text_input_ids.shape[1]:, -text_input_ids.shape[1]:] # get the text part of the attention map

            attn1 = attention_map1.flatten()
            attn2 = attention_map2.flatten()
            similarity = cosine_similarity(attn1, attn2)
            hal_similarities.append(similarity)

            visualize_attention_compare([attention_map1, attention_map2], layer_idx, head_idx, \
                                        output_path = os.path.join(args.output_path, f"{question_id}_halhead_{layer_idx}_{head_idx}_top{idx}.png"), \
                                        title="Hallucination Head (Layer {}, Head {}), Sim: {:.2f}".format(layer_idx, head_idx, similarity.item()), \
                                        stride=stride)

        print(hal_similarities)
        hal_similarities = torch.tensor(hal_similarities).mean()

        print('-----')
        non_hal_similarities = []
        for idx, attn_head in enumerate(non_hal_attention_heads):
            layer_idx, head_idx = attn_head
            attention_map1 = output_attentions1[layer_idx][0][0][head_idx].cpu().float()
            attention_map1 = attention_map1[-text_input_ids.shape[1]:, -text_input_ids.shape[1]:] # get the text part of the attention map
            
            attention_map2 = output_attentions2[layer_idx][0][head_idx].cpu().float()
            attention_map2 = attention_map2[-text_input_ids.shape[1]:, -text_input_ids.shape[1]:] # get the text part of the attention map

            attn1 = attention_map1.flatten()
            attn2 = attention_map2.flatten()
            similarity = cosine_similarity(attn1, attn2)
            non_hal_similarities.append(similarity)

            visualize_attention_compare([attention_map1, attention_map2], layer_idx, head_idx, \
                                        output_path = os.path.join(args.output_path, f"{question_id}_nonhalhead_{layer_idx}_{head_idx}_top{idx}.png"), \
                                        title="Non-Hallucination Head (Layer {}, Head {}), Sim: {:.2f}".format(layer_idx, head_idx, similarity.item()), \
                                        stride=stride)
        
        print(non_hal_similarities)
        non_hal_similarities = torch.tensor(non_hal_similarities).mean()
        
        hal_similarities_all_samples.append(hal_similarities)
        non_hal_similarities_all_samples.append(non_hal_similarities)

        torch.cuda.empty_cache()
        
        if count >= 100:
            break
    
    torch.save(hal_similarities_all_samples, os.path.join(args.output_path, 'hal_similarities_all_samples.pth'))
    torch.save(non_hal_similarities_all_samples, os.path.join(args.output_path, 'non_hal_similarities_all_samples.pth'))
    hal_similarities_avg = torch.mean(torch.tensor(hal_similarities_all_samples))
    non_hal_similarities_avg = torch.mean(torch.tensor(non_hal_similarities_all_samples))
    print(f'Average similarity for hal heads: {hal_similarities_avg}')
    print(f'Average similarity for non-hal heads: {non_hal_similarities_avg}')
    

def plot_results(args):

    hal_similarities_all_samples = torch.load(os.path.join(args.output_path, 'hal_similarities_all_samples.pth'))
    non_hal_similarities_all_samples = torch.load(os.path.join(args.output_path, 'non_hal_similarities_all_samples.pth'))
    
    hal_similarities_avg = torch.mean(torch.tensor(hal_similarities_all_samples))
    non_hal_similarities_avg = torch.mean(torch.tensor(non_hal_similarities_all_samples))
    
    fig, ax = plt.subplots()
    width = 0.2
    spacing = -0.6
    categories = ['Hallucination Heads', 'Non-Hallucination Heads']
    
    x = np.arange(len(categories)) 
    x_positions = x + spacing * np.array([0, 1])
    attention_similarity = [hal_similarities_avg, non_hal_similarities_avg]
    print(attention_similarity)
    ax.bar(x_positions, attention_similarity, width=width, color=['skyblue', 'lightslategray'],  tick_label=categories)

    ax.set_ylabel('Cosine Similarity', fontsize=16)
    ax.set_ylim(0.65, 0.8)  
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.7)  

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_path, 'attention_similarity.png'), bbox_inches='tight')


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
    parser.add_argument("--top_k", type=int, default=1)
    args = parser.parse_args()
    set_seed(args.seed)
    
    eval_model(args)
    plot_results(args)
