export HF_HUB_OFFLINE=True

SEED=42
DATE=`date "+%m%d"`

# model_name=llava-1.5 # or minigpt4
model_name=minigpt4
max_new_tokens=128
num_samples=500
data_path=../dataset
method=adhh
adhh_threshold=0.5
top_k=20 # for llava-1.5, we use top_k=20, for minigpt4, we use top_k=10

## coco dataset
dataset=coco
EXP=${model_name}_${method}_topk${top_k}_threshold${adhh_threshold}_token${max_new_tokens}_${SEED}_n${num_samples}
result_path=./results/$dataset/$DATE/$EXP
echo $result_path

CUDA_VISIBLE_DEVICES='3' python -m eval_scripts.eval_caption_adhh \
--model $model_name \
--dataset_name $dataset \
--image_folder $data_path/coco/val2014 \
--caption_file_path $data_path/coco/annotations/captions_val2014.json \
--seed $SEED \
--max_new_tokens $max_new_tokens \
--num_samples $num_samples \
--output_dir $result_path \
--attention_head_path ./results/${model_name}_attribution_result.json \
--top_k $top_k \
--adaptive_deactivate \
--deactivate_threshold $adhh_threshold

python eval_scripts/eval_utils/eval_chair.py \
    --annotation-dir $data_path/coco/annotations \
    --answers-file $result_path/captions.jsonl \
    --caption_file captions_val2014.json

## nocaps dataset
dataset=nocaps
EXP=${model_name}_${method}_topk${top_k}_threshold${adhh_threshold}_token${max_new_tokens}_${SEED}_n${num_samples}
result_path=./results/$dataset/$DATE/$EXP
echo $result_path

CUDA_VISIBLE_DEVICES='3' python -m eval_scripts.eval_caption_adhh \
--model $model_name \
--dataset_name $dataset \
--image_folder $data_path/nocaps/images/val \
--caption_file_path $data_path/nocaps/annotations/nocaps_val_4500_captions.json \
--seed $SEED \
--max_new_tokens $max_new_tokens \
--num_samples $num_samples \
--output_dir $result_path \
--attention_head_path ./results/${model_name}_attribution_result.json \
--top_k $top_k \
--adaptive_deactivate \
--deactivate_threshold $adhh_threshold

python eval_scripts/eval_utils/eval_nocaps_chair.py \
--annotation-dir $data_path/nocaps/images/val \
--answers-file $result_path/captions.jsonl \
--caption_file nocaps_val_4500_captions.json
