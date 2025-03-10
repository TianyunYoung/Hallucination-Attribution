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

from decoder_zoo.HALC.context_density.halc import halc_assistant
from decoder_zoo.VCD.vcd_utils.vcd_add_noise import add_diffusion_noise
from transformers import StoppingCriteriaList, MaxLengthCriteria

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


parser = argparse.ArgumentParser(description="Evaluation on LVLMs.")
parser.add_argument("--model", type=str, default="minigpt4", help="model")
parser.add_argument("--merged_ckpt", type=str, default=None)
parser.add_argument("--decoder", type=str, default="greedy", help="Decoding strategy to use. You can choose from 'greedy', 'vcd', 'opera', 'dola', 'halc'. Default is 'greedy'.")
parser.add_argument("--beam", type=int, default=1)
parser.add_argument("--sample", action="store_true")
parser.add_argument("--max_new_tokens", type=int, default=64)
parser.add_argument("--num_samples", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--options", nargs="+", help="override some settings in the used config, the key-value pair " "in xxx=yyy format will be merged into config file (deprecate), ""change to --cfg-options instead.",)
# data
parser.add_argument("--dataset_name", type=str, default="coco", help="Name of the dataset. Default is 'coco'.")
parser.add_argument("--image_folder", type=str, default="eval_dataset/val2014/", help="data path",)
parser.add_argument("--caption_file_path", type=str, default="coco", help="Name of the dataset. Default is 'coco'.")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--num_workers", type=int, default=2, help="num workers")
parser.add_argument("--output_dir", type=str, default="./log/", help="Output ditectory for saving test results. Default is './generated_chair_inputs/'.")
parser.add_argument("--verbosity", action="store_false", dest="verbosity", default=True, help="Verbosity. Default: True.")
# opera
parser.add_argument("--scale_factor", type=float, default=50)
parser.add_argument("--threshold", type=int, default=15)
parser.add_argument("--num_attn_candidates", type=int, default=5)
parser.add_argument("--penalty_weights", type=float, default=1.0)
# vcd
parser.add_argument("--cd_alpha", type=float, default=1, help="Alpha param for VCD.")
parser.add_argument("--cd_beta", type=float, default=0.1, help="Beta param for VCD.")
parser.add_argument("--noise_step", type=int, default=500, help="Noise step for VCD.")
# halc
parser.add_argument("--detector", type=str, default="dino", help="Detector type. Default is 'groundingdino'.")
parser.add_argument("--k-candidate-num", type=int, default=4, help="specify the k candidate number for halc.")
parser.add_argument("--expand-ratio", type=float, default=0.6, help="Expand ratio of growing contextual field.")
parser.add_argument("--box_threshold", type=float, default=0.4, help="Box threshold for DINO.")
parser.add_argument("--debugger", type=float, default=0)

args = parser.parse_known_args()[0]

device = torch.device("cuda:0")

# ========================================
#             Argument Initialization
# ========================================
model_name = args.model
# decoding params
decoding_strategy = args.decoder
num_beams = args.beam
sample = args.sample
max_new_tokens = args.max_new_tokens
output_dir = args.output_dir
verbosity = args.verbosity
# dataset params
dataset_name = args.dataset_name
image_folder = args.image_folder
num_workers = args.num_workers
batch_size = args.batch_size
num_samples = args.num_samples
# halc params
detector_type = args.detector
box_threshold = args.box_threshold
k_candidate_num = args.k_candidate_num
expand_ratio = args.expand_ratio
debugger = args.debugger
# vcd params
noise_step = args.noise_step
cd_alpha = args.cd_alpha
cd_beta = args.cd_beta
# dola params
lm_early_exit_layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
mature_layer = lm_early_exit_layers[-1]
premature_layer = None
candidate_premature_layers = lm_early_exit_layers[:-1]
premature_layer_dist = {l: 0 for l in candidate_premature_layers}
# opera params
scale_factor=args.scale_factor
threshold=args.threshold
num_attn_candidates=args.num_attn_candidates
penalty_weights=args.penalty_weights

# ========================================
#             Model Initialization
# ========================================
args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
cfg = Config(args)
seed = args.seed
setup_seeds(cfg, seed)

model_config = cfg.model_cfg
# model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
model.eval()

processor_cfg = cfg.get_config().preprocess
processor_cfg.vis_processor.eval.do_normalize = False
vis_processors, txt_processors = load_preprocess(processor_cfg)

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

os.makedirs(args.output_dir, exist_ok=True)
save_args_to_json(args, os.path.join(args.output_dir, 'config.json'))

# ========================================
#    Initializing decoding strategy
# ========================================
valid_decoding_strategies = [
    "greedy",
    "dola",
    "halc",
    "opera",
    "vcd",
    "beam"
]

valid_detector = ["dino", "owlv2"]
assert (
    decoding_strategy in valid_decoding_strategies
), f"Invalid decoding strategy: {decoding_strategy}, should be in {valid_decoding_strategies}"
assert (
    detector_type in valid_detector
), f"Invalid detector type: {detector_type}, should be in {valid_detector}"

decoding_strategy = decoding_strategy
opera_decoding = False
dola_decoding = False
halc_decoding = False
vcd_decoding = False
beam_search = False

stopping_criteria = None
output_attentions = False
if decoding_strategy == "greedy":
    pass
elif decoding_strategy == "dola":
    dola_decoding = True
elif decoding_strategy == "halc":
    halc_decoding = True
    dola_decoding = True
    beam_search = True
elif decoding_strategy == "opera":
    beam_search = True
    opera_decoding = True
    num_beams = 5
    output_attentions = True
    stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=args.max_new_tokens)])
elif decoding_strategy == "beam":
    beam_search = True
elif decoding_strategy == "vcd":
    vcd_decoding = True

print(f"\033[42m####### Current Decoding Strategy: {decoding_strategy} #######\033[0m")

if verbosity:
    print("\ndecoding strategy: ", decoding_strategy)
    print("backbone model_name: ", args.model)
    print("dataset_name: ", dataset_name)
    print("image_folder: ", image_folder)
    print("output_dir: ", output_dir)
    print("num_samples: ", num_samples)
    print("num_beams: ", num_beams)
    print("seed: ", seed)
    print(vis_processors["eval"].transform)

halc_params = {
    "context_domain": "upper",
    "contrast_weight": 0.05,
    "context_window": 4,
    "expand_ratio": expand_ratio,
    "beam_size": num_beams,
    "k_candidate_num": k_candidate_num,
    "LVLM_backbone": model_name,
    "detector": detector_type,
    "score_type": "BLIP",
    "debugger": debugger,
    "box_threshold": box_threshold,
}

halc_assistant_helper = halc_assistant(
    model,
    vis_processor=vis_processor,
    device=device,
    halc_params=halc_params,
    max_new_tokens=max_new_tokens,
)

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
    answers_file = os.path.join(args.output_dir, 'captions.jsonl')
elif dataset_name == 'mmvet':
    answers_file = os.path.join(args.output_dir, 'captions.json')
    result_dict = {}

# ========================================
#    Generating Answers
# ========================================
ans_file = open(answers_file, "w")
start_time = time.time()
for line in tqdm(questions, total=len(questions)):
    question_id = line["question_id"]
    cur_prompt = line["text"]
    image_file = line["image"]

    image_path = os.path.join(image_folder, image_file)
    raw_image = Image.open(image_path).convert('RGB')

    template = INSTRUCTION_TEMPLATE[args.model]
    prompt = template.replace("<question>", cur_prompt)
    image = vis_processors["eval"](raw_image).unsqueeze(0)
    image = image.to(device)

    halc_assistant_helper.update_input(img_path=image_path, input_prompt=prompt)
    image_cd = None

    # vcd_decoding
    if vcd_decoding:
        image_tensor_cd = add_diffusion_noise(image, noise_step)
        image_cd = (
            image_tensor_cd.unsqueeze(0).half().cuda()
            if image_tensor_cd is not None
            else None
        )
        if model_name == "minigpt4":
            image_cd = image_cd.squeeze(0)

    # generate
    with torch.inference_mode():
        with torch.no_grad():
            out = model.generate(
                {"image": norm(image), "prompt": prompt, "img_path": image_path},
                output_attentions=output_attentions,
                # Decoding
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_nucleus_sampling=sample,
                beam_search=beam_search,
                dola_decoding=dola_decoding,
                opera_decoding=opera_decoding,
                vcd_decoding=vcd_decoding,
                halc_decoding=halc_decoding,
                # DOLA
                premature_layer=premature_layer,
                candidate_premature_layers=candidate_premature_layers,
                mature_layer=mature_layer,
                # OPERA
                key_position=None,
                scale_factor=scale_factor,
                threshold=threshold,
                num_attn_candidates=num_attn_candidates,
                penalty_weights=penalty_weights,
                # VCD
                images_cd=image_cd,
                cd_alpha=cd_alpha,
                cd_beta=cd_beta,
                # HALC
                halc_assistant=halc_assistant_helper,
            )

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
    if dataset_name == 'coco' or dataset_name == 'nocaps' or dataset_name == 'mme':
        ans_file.write(json.dumps({"question_id": question_id,
                                    "image": image_file,
                                    "prompt": cur_prompt,
                                    "text": output_text,
                                    "model_id": model_name}) + "\n")
        ans_file.flush()
    elif dataset_name == 'mmvet':
        result_dict[question_id] = output_text

if dataset_name == 'mmvet':
    with ans_file as f:
        json.dump(result_dict, f, indent=4)

end_time = time.time()
print(decoding_strategy, end_time - start_time)
