import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import json
import time
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from minigpt4.models import load_preprocess
from minigpt4.common.config import Config
from minigpt4.common.registry import registry

from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from pycocotools.coco import COCO

MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "llava-1.5": "USER: <ImageHere>\n<question> ASSISTANT:",
}

def setup_seeds(config, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def save_args_to_json(args, config_filename='config.json'):
    args_dict = vars(args)
    with open(config_filename, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

parser = argparse.ArgumentParser(description="caption evaluation on LVLMs.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--model", type=str, default="minigpt4", help="model")
parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")

parser.add_argument("--caption_file_path", type=str, default="coco", help="Name of the dataset. Default is 'coco'.")
parser.add_argument("--dataset_name", type=str, default="coco", help="Name of the dataset. Default is 'coco'.")
parser.add_argument("--image_folder", type=str, default="eval_dataset/val2014/", help="data path",)
parser.add_argument("--num_samples", type=int, default=100)
parser.add_argument("--output_dir", type=str, default="./log/", help="Output ditectory for saving test results. Default is './generated_chair_inputs/'.")
parser.add_argument("--options", nargs="+", help="override some settings in the used config, the key-value pair " "in xxx=yyy format will be merged into config file (deprecate), ""change to --cfg-options instead.",)

parser.add_argument("--beam", type=int, default=1)
parser.add_argument("--max_new_tokens", type=int, default=128)
parser.add_argument("--attention_head_path", type=str, default='')
parser.add_argument("--top_k", type=int, default=30)
parser.add_argument("--adaptive_deactivate", action='store_true', default=False)
parser.add_argument("--deactivate_threshold", type=float, default=0.5)
parser.add_argument("--merged_ckpt", type=str, default=None)
args = parser.parse_known_args()[0]

# device, model, seed setup
device = torch.device("cuda:0")

# args init
model_name = args.model
dataset_name = args.dataset_name
image_folder = args.image_folder
output_dir = args.output_dir
num_beams = args.beam
max_new_tokens = args.max_new_tokens
os.makedirs(output_dir, exist_ok=True)

# ========================================
#             Model Initialization
# ========================================
args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
cfg = Config(args)
seed = args.seed
setup_seeds(cfg, seed)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
model.eval()

processor_cfg = cfg.get_config().preprocess
processor_cfg.vis_processor.eval.do_normalize = False
vis_processors, txt_processors = load_preprocess(processor_cfg)

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

# ========================================
#    Initializing dataset
# ========================================
mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)

if dataset_name == 'coco':
    caption_file_path = args.caption_file_path
    coco = COCO(caption_file_path)
    img_ids = coco.getImgIds()
    sampled_img_ids = random.sample(img_ids, args.num_samples)

    questions = []
    for sampled_img_id in sampled_img_ids:
        image_file = coco.loadImgs(sampled_img_id)[0]["file_name"]
        question = {
            "question_id": sampled_img_id,
            "image": image_file,
            "text": "Please describe this image in detail.",
        }
        questions.append(question)

elif dataset_name == 'nocaps':  
    caption_file_path = args.caption_file_path
    val_caps = json.load(open(caption_file_path))
    image_infos = val_caps["images"]
    out_image_infos = [image_info for image_info in image_infos if image_info['domain'] == 'out-domain']
    sampled_img_infos = random.sample(out_image_infos, args.num_samples)

    questions = []
    for sampled_img_info in sampled_img_infos:
        image_file = sampled_img_info['file_name']
        image_id = sampled_img_info['id']
        question = {
            "question_id": sampled_img_info['id'],
            "image": sampled_img_info['file_name'],
            "text": "Please describe this image in detail.",
        }
        questions.append(question)
elif dataset_name == 'mmvet':
    json_infos = json.load(open(args.caption_file_path, "r"))
    questions = []
    for key, value in json_infos.items():
        question = {
            "question_id": key,
            "image": value['imagename'],
            "text": value['question'],
        }
        questions.append(question)
elif dataset_name == 'mme':
    questions = []
    for idx, q in enumerate(open(args.caption_file_path)):
        question = json.loads(q) 
        question = {
            "question_id": question['question_id'],
            "image": question['image'],
            "text": question['text'],
        }
        questions.append(question)

if dataset_name == 'coco' or dataset_name == 'nocaps' or dataset_name == 'mme':
    answers_file = os.path.join(args.output_dir, 'answers.jsonl')
elif dataset_name == 'mmvet':
    answers_file = os.path.join(args.output_dir, 'answers.json')
    result_dict = {}

print(len(questions))

# Read data from the file
with open(args.attention_head_path, 'r') as file:
    data_loaded = json.load(file)

args.hal_attention_heads = data_loaded['hal_heads']
hal_attention_heads = args.hal_attention_heads[:args.top_k]
save_args_to_json(args, os.path.join(output_dir, 'config.json'))

answers_file = os.path.join(output_dir, f'captions.jsonl')
ans_file = open(answers_file, "w")
for line in tqdm(questions, total=len(questions)):
    question_id = line["question_id"]
    cur_prompt = line["text"]
    image_file = line["image"]

    image_path = os.path.join(image_folder, image_file)
    print(image_path)
    raw_image = Image.open(image_path).convert('RGB')

    template = INSTRUCTION_TEMPLATE[args.model]
    prompt = template.replace("<question>", cur_prompt)
    image = vis_processors["eval"](raw_image).unsqueeze(0)
    image = image.to(device)

    with torch.inference_mode():
        with torch.no_grad():
            out = model.generate(
                {"image": norm(image), "prompt": prompt, "img_path": image_path},
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                output_attentions=False,
                hal_attention_heads=hal_attention_heads,
                adaptive_deactivate=args.adaptive_deactivate,
                deactivate_threshold=args.deactivate_threshold)

    output_text = out[0]
    print(question_id, output_text)
    
    # remove unk
    sentence_list = output_text.split(".")
    sentence_filter_list = []
    for sentence in sentence_list:
        if "unk" not in sentence:
            sentence_filter_list.append(sentence)
    output_text = ".".join(sentence_filter_list)

    # save results
    ans_file.write(json.dumps({"question_id": question_id,
                                "image": image_file,
                                "prompt": cur_prompt,
                                "text": output_text,
                                "model_id": model_name}) + "\n")
    ans_file.flush()

