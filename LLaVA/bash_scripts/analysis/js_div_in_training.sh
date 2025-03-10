export HF_HUB_OFFLINE=True

model_path=liuhaotian/llava-v1.5-7b
data_path=../dataset
result_path=./results/coco/llava_3000

# Figure 5: analyze the JS divergence of the attention map from the initial model throughout the instruction tuning process.
for step in 500 1000 1500 2000 2500 3000 3500 4000 4500 5000
do
model_path=../checkpoints/llava-v1.5-7b-instruction-tuning/checkpoint-${step}
CUDA_VISIBLE_DEVICES='0,1' python -m eval_scripts.analyze_js_div_in_training \
    --model-path $model_path \
    --image-folder $data_path/coco/train2014 \
    --question-file $result_path/captions_eval_results.json \
    --output-path $result_path/js_div_in_training/step${step} \
    --attention_head_path $result_path/identify_attention_head/attribution_result.json \
    --temperature 0 \
    --conv-mode vicuna_v1
done

python -m eval_scripts.plot_js_div_in_training \
    --output-path $result_path/js_div_in_training \
    --step-list 500 1000 1500 2000 2500 3000 3500 4000 4500 5000
