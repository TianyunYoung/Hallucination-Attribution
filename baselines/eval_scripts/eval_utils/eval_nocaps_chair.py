import os
import argparse
import json
import nocaps_chair
import numpy as np
import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b", use_fast=False)
from eval_bleu import Evaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--annotation-dir", type=str, default="")
    parser.add_argument("--domain", type=str, default="out-domain")
    parser.add_argument("--caption_file", type=str, default="nocaps_val_4500_captions.json")
    args = parser.parse_args()

    answers = []
    for line in open(args.answers_file):
        ans = json.loads(line)
        answer = {
             "caption": ans['text'],
             "image": ans['image'],
             "image_id": os.path.splitext(ans['image'])[0],
             "question_id": ans['question_id'],
        }
        answers.append(answer)
    
    imids = [answer['image_id'] for answer in answers]
    outputs = [i["caption"] for i in answers]
    
    # initialize CHAIR with generated captions and annotations
    evaluator = nocaps_chair.CHAIR(imids, args)
    
    # compute chair metrics
    cap_dict = evaluator.compute_chair(answers)

    # get gt captions
    caption_file_path = os.path.join(args.annotation_dir, args.caption_file)
    caption_json = json.load(open(caption_file_path))
    caption_annot = caption_json['annotations']

    caption_dict = {k:[] for k in range(4500)}
    for caption_annot in caption_annot:
        caption_dict[caption_annot['image_id']].append(caption_annot['caption'])
    captions = []
    sampled_img_ids = [answer["question_id"] for answer in answers]
    for sampled_img_id in sampled_img_ids:
        captions.append(caption_dict[sampled_img_id])
    
    # calculate metrics
    golden_reference = []
    candidate_reference = []
    for i, caption in enumerate(captions):
        candidate_reference.append(outputs[i])
        golden_reference.append(caption)

    golden_reference = {k: [{'caption': x} for x in v] for k, v in enumerate(golden_reference)}
    candidate_reference = {k: [{'caption': v}] for k, v in enumerate(candidate_reference)}

    evaluator = Evaluator()
    evaluator.do_the_thing(golden_reference, candidate_reference)
    results = evaluator.evaluation_report

    # save to json pretty print
    chair_json_path = args.answers_file.replace('.jsonl', '_eval_results.json')
    assert chair_json_path != args.answers_file
    
    with open(chair_json_path, "w") as f:
        json.dump(cap_dict, f, indent=4)

    print(cap_dict['overall_metrics'])


