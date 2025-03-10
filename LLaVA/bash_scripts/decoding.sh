export HF_HUB_OFFLINE=True

model_name=llava-v1.5-7b
model_path=liuhaotian/llava-v1.5-7b
# model_name=llava-v1.5-13b
# model_path=liuhaotian/llava-v1.5-13b
# model_name=llava-v1.6-34b
# model_path=liuhaotian/llava-v1.6-34b
dataset=coco
data_path=../dataset
num_samples=500
adhh_threshold=0.4
result_path=./results/$dataset/${model_name}_threshold${adhh_threshold}_n${num_samples}

CUDA_VISIBLE_DEVICES='1' python -m eval_scripts.eval_caption_adhh \
--model-path $model_path \
--image-folder $data_path/coco/val2014 \
--caption_file_path $data_path/coco/annotations/captions_val2014.json \
--answers-file $result_path/captions.jsonl \
--dataset $dataset \
--temperature 0 \
--conv-mode vicuna_v1 \
--num_samples $num_samples \
--adaptive_deactivate \
--adhh_threshold $adhh_threshold

python eval_scripts/eval_utils/eval_chair.py \
    --annotation-dir $data_path/coco/annotations \
    --answers-file $result_path/captions.jsonl \
    --caption_file captions_val2014.json

