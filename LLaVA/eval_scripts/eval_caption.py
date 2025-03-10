import os
import json
import argparse
import shortuuid
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

import torch
from torch.utils.data import Dataset, DataLoader

import math
import shutil
import random
from PIL import Image
from transformers import set_seed
from pycocotools.coco import COCO

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

    if args.dataset == 'coco':
        caption_file_path = args.caption_file_path
        coco = COCO(caption_file_path)
        img_ids = coco.getImgIds()
        sampled_img_ids = random.sample(img_ids, args.num_samples)

        questions = []
        dest_image_folder = os.path.join(os.path.split(os.path.split(os.path.dirname(args.answers_file))[0])[0], 'images', f'seed{args.seed}_{args.num_samples}')
        os.makedirs(dest_image_folder, exist_ok=True)
        for sampled_img_id in sampled_img_ids:
            image_file = coco.loadImgs(sampled_img_id)[0]["file_name"]
            question = {
                "question_id": sampled_img_id,
                "image": image_file,
                "text": "Please describe this image in detail.",
            }
            shutil.copyfile(os.path.join(args.image_folder, image_file), os.path.join(dest_image_folder, image_file))
            questions.append(question)

    elif args.dataset == 'nocaps':  
        caption_file_path = args.caption_file_path
        val_caps = json.load(open(caption_file_path))
        image_infos = val_caps["images"]
        out_image_infos = [image_info for image_info in image_infos if image_info['domain'] == 'out-domain']
        sampled_img_infos = random.sample(out_image_infos, args.num_samples)

        questions = []
        dest_image_folder = os.path.join(os.path.split(os.path.split(os.path.dirname(args.answers_file))[0])[0], 'images', f'seed{args.seed}_{args.num_samples}')
        os.makedirs(dest_image_folder, exist_ok=True)
        for sampled_img_info in sampled_img_infos:
            image_file = sampled_img_info['file_name']
            image_id = sampled_img_info['id']
            question = {
                "question_id": sampled_img_info['id'],
                "image": sampled_img_info['file_name'],
                "text": "Please describe this image in detail.",
            }
            shutil.copyfile(os.path.join(args.image_folder, image_file), os.path.join(dest_image_folder, f'{image_id}_{image_file}'))
            questions.append(question)

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        question_id = line["question_id"]
        cur_prompt = line["text"]
        image_file = line["image"]


        input_ids = input_ids.to(device='cuda', non_blocking=True)
        image_tensor = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)
        
        with torch.inference_mode():
            output_dict = model.generate(
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
                output_scores=True,
                output_hidden_states=True,
                return_dict_in_generate=True)

        output_ids = output_dict['sequences']
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(question_id, outputs)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": question_id,
                                "image": image_file,
                                "prompt": cur_prompt,
                                "text": outputs,
                                "answer_id": ans_id,
                                "model_id": model_name,
                                "metadata": {}}) + "\n")
        ans_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--caption_file_path", type=str, default="")
    parser.add_argument("--question-file", type=str, default="question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answers.jsonl")
    parser.add_argument("--dataset", type=str, default="coco")
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

    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
