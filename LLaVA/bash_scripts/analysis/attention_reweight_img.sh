export HF_HUB_OFFLINE=True

model_path=liuhaotian/llava-v1.5-7b
data_path=../dataset

# Figure 6(b): analyze the effect of upscaling the image attention of hallucination heads
for reweight_alpha in 1.0 1.5 2.0 2.5 3.0 3.5
do
dataset=coco
top_k=20
result_path=./results/$dataset/halheads_reweight_img_top${top_k}_${reweight_alpha}

CUDA_VISIBLE_DEVICES='0' python -m eval_scripts.analyze_attention_reweight \
    --model-path $model_path \
    --image-folder $data_path/coco/val2014 \
    --caption_file_path $data_path/coco/annotations/captions_val2014.json \
    --answers-file $result_path/captions.jsonl \
    --dataset $dataset \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --attention_head_path ./results/coco/llava_3000/identify_attention_head/attribution_result.json \
    --top_k $top_k \
    --reweight_img \
    --reweight_alpha $reweight_alpha

python eval_scripts/eval_utils/eval_chair.py \
    --annotation-dir $data_path/coco/annotations \
    --answers-file $result_path/captions.jsonl \
    --caption_file captions_val2014.json
done
