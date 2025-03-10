from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import pandas as pd
import json
import sys
import argparse

class Evaluator:
    def __init__(self) -> None:
        self.tokenizer = PTBTokenizer()
        self.scorer_list = [
            (Cider(), "CIDEr"),
            (Bleu(), "Bleu"),
            (Rouge(), "Rough"),
            (Meteor(), "Meteor"),
        ]
        self.evaluation_report = {}

    def do_the_thing(self, golden_reference, candidate_reference):
        golden_reference = self.tokenizer.tokenize(golden_reference)
        candidate_reference = self.tokenizer.tokenize(candidate_reference)
        
        # From this point, some variables are named as in the original code
        # I have no idea why they name like these
        # The original code: https://github.com/salaniz/pycocoevalcap/blob/a24f74c408c918f1f4ec34e9514bc8a76ce41ffd/eval.py#L51-L63
        for scorer, method in self.scorer_list:
            score, scores = scorer.compute_score(golden_reference, candidate_reference)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    self.evaluation_report[m] = sc
            else:
                self.evaluation_report[method] = score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--dataset", type=str, default="coco")
    args = parser.parse_args()

    if args.dataset == 'nocaps':
        val_caps = json.load(open(args.question_file))
        ann_infos = val_caps["annotations"]
        
        outputs = [json.loads(x)['text'] for x in open(args.answers_file).readlines()]
        image_ids = [json.loads(x)['question_id'] for x in open(args.answers_file).readlines()]

        ann_infos_chunk = [ann_infos[i:i+10] for i in range(0, len(ann_infos), 10)]
        ann_infos_chunk_resort = []
        for image_id in image_ids:
            ann_infos_chunk_resort.append(ann_infos_chunk[image_id])

        captions = []
        for idx, ann_infos in enumerate(ann_infos_chunk_resort):
            caption = [ann_info['caption'] for ann_info in ann_infos]
            captions.append(caption)

        import pdb; pdb.set_trace()   
    else:
        f = open(args.question_file)
        captions = [json.loads(x)['caption'] for x in f.readlines()]

        f = open(args.answers_file)
        outputs = [json.loads(x)['text'] for x in f.readlines()]
    

    golden_reference = []
    candidate_reference = []
    for i, caption in enumerate(captions):
        golden_reference.append(caption)
        candidate_reference.append(outputs[i])

    golden_reference = {k: [{'caption': x} for x in v] for k, v in enumerate(golden_reference)}
    candidate_reference = {k: [{'caption': v}] for k, v in enumerate(candidate_reference)}

    evaluator = Evaluator()
    evaluator.do_the_thing(golden_reference, candidate_reference)

    print(evaluator.evaluation_report)
