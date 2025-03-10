import os
import argparse
import json
import chair
import numpy as np
import transformers
from eval_bleu import Evaluator
from pycocotools.coco import COCO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--annotation-dir", type=str, default="answer.jsonl")
    parser.add_argument("--caption_file", type=str, default="captions_train2014.json")
    args = parser.parse_args()

    answers = []
    for line in open(args.answers_file):
        ans = json.loads(line)
        answer = {
             "caption": ans['text'],
             "image": ans['image'],
             "image_id": ans['question_id'],
        }
        answers.append(answer)

    imids = [answer['image_id'] for answer in answers]

    # initialize CHAIR with generated captions and annotations
    evaluator = chair.CHAIR(imids, args.annotation_dir)
    evaluator.get_annotations()
    
    # compute chair metrics
    cap_dict = evaluator.compute_chair(answers)
    caption_result = cap_dict["sentences"]
    outputs = [i["caption"] for i in caption_result]

    # get gt captions
    caption_file_path = os.path.join(args.annotation_dir, args.caption_file)
    coco = COCO(caption_file_path)
    sampled_img_ids = [i["image_id"] for i in caption_result]
    captions = []
    for sampled_img_id in sampled_img_ids:
        ann_ids = coco.getAnnIds(imgIds=sampled_img_id)
        annotations = coco.loadAnns(ann_ids)
        captions.append([ann['caption'] for ann in annotations])

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


