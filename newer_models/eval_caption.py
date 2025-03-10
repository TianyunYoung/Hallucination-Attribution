import os
import json
import random
import shutil
import argparse
from tqdm import tqdm
from PIL import Image
import torch
from transformers import set_seed
from pycocotools.coco import COCO
from transformers import MllamaForConditionalGeneration, AutoProcessor
from transformers import ChameleonForConditionalGeneration, ChameleonProcessor

def eval_model(args):

    checkpoint = args.model_path
    if checkpoint == 'meta-llama/Llama-3.2-11B-Vision-Instruct':
        model = MllamaForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.float16, device_map='auto', _attn_implementation='eager')
        processor = AutoProcessor.from_pretrained(checkpoint)
    elif checkpoint == 'facebook/chameleon-7b':
        model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", torch_dtype=torch.bfloat16, device_map="cuda", _attn_implementation='eager')
        processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")
    elif checkpoint == 'facebook/chameleon-30b':
        model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-30b", torch_dtype=torch.bfloat16, device_map="cuda", _attn_implementation='eager')
        processor = ChameleonProcessor.from_pretrained("facebook/chameleon-30b")

    os.makedirs(os.path.split(args.answers_file)[0], exist_ok=True)
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

    questions = questions[args.start_idx:args.end_idx]
    ans_file = open(args.answers_file, "w")

    if checkpoint == 'meta-llama/Llama-3.2-11B-Vision-Instruct':
        hal_attention_heads = [[33, 3], [27, 11], [28, 3], [32, 15], [22, 27], [22, 29], [38, 3], [20, 29], [26, 14], [26, 12]]
    elif checkpoint == 'facebook/chameleon-7b':
        hal_attention_heads = [[25, 2], [25, 11], [26, 20], [24, 19], [22, 12], [23, 17], [20, 24], [22, 23], [18, 16], [29, 15]]
    elif checkpoint == 'facebook/chameleon-30b':
        hal_attention_heads = [[40, 40], [40, 45], [36, 52], [40, 47], [34, 39], [36, 31], [34, 33], [32, 6], [36, 60], [40, 44]]
    hal_attention_heads = hal_attention_heads[:args.top_k]

    if args.prune:
        from intervention_decoding import set_new_sample
        set_new_sample()
    
    for line in tqdm(questions, total=len(questions)):
        question_id = line["question_id"]
        prompt = line["text"]
        image_file = line["image"]
        image_path = os.path.join(args.image_folder, image_file)
        image = Image.open(image_path)

        if checkpoint == 'facebook/chameleon-7b' or checkpoint == 'facebook/chameleon-30b':
            prompt = prompt + "<image>"
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
            if args.prune:
                output = model.generate(**inputs, \
                                        do_sample=False, \
                                        head_dim=128, \
                                        max_new_tokens=args.max_new_tokens, \
                                        hal_attention_heads=hal_attention_heads)
            
            else:
                output = model.generate(**inputs, \
                                        max_new_tokens=128, \
                                        do_sample=False)
            decoded_output = processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
            print(decoded_output)
        else:
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]}
                ]
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)

            if args.prune:
                output = model.generate(
                                        **inputs, \
                                        do_sample=False,
                                        head_dim=128, \
                                        max_new_tokens=args.max_new_tokens,
                                        hal_attention_heads=hal_attention_heads)

            else:
                output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                
            decoded_output = processor.decode(output[0][inputs.input_ids.shape[1]:])
            print(decoded_output)

        ans_file.write(json.dumps({"question_id": question_id,
                            "image": image_file,
                            "prompt": line["text"],
                            "text": decoded_output,
                            "model_id": args.model_path}) + "\n")
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--caption_file_path", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="answers.jsonl")
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=10000)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--prune", action="store_true", default=False)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
