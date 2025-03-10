export HF_HUB_OFFLINE=True

dataset=coco
num_samples=3000
model_path=liuhaotian/llava-v1.5-7b
data_path=../dataset
result_path=./results/$dataset/llava_${num_samples}

# generate captions for COCO training set
CUDA_VISIBLE_DEVICES='0' python -m eval_scripts.eval_caption \
    --model-path $model_path \
    --image-folder $data_path/coco/train2014 \
    --caption_file_path $data_path/coco/annotations/captions_train2014.json \
    --answers-file $result_path/captions.jsonl \
    --dataset $dataset \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --num_samples $num_samples

# evaluate on generated captions, extract hallucination and non-hallucination objects
python eval_scripts/eval_utils/eval_chair.py \
    --annotation-dir $data_path/coco/annotations \
    --answers-file $result_path/captions.jsonl

# identify the hallucination heads
CUDA_VISIBLE_DEVICES='0' python -m eval_scripts.identify_attention_head \
    --model-path $model_path \
    --image-folder $data_path/coco/train2014 \
    --output-path $result_path/identify_attention_head \
    --answers-file $result_path/captions_eval_results.json \
    --temperature 0 \
    --conv-mode vicuna_v1
