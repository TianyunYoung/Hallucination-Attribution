
export HF_HUB_OFFLINE=True

model_path=liuhaotian/llava-v1.5-7b
data_path=../dataset
result_path=./results/coco/llava_3000

# Figure 4: analyze the inheritance of attention patterns from base language models 
CUDA_VISIBLE_DEVICES='0,1' python -m eval_scripts.analyze_attention_inheritance \
    --model-path $model_path \
    --image-folder $data_path/coco/train2014 \
    --answers-file $result_path/captions_eval_results.json \
    --attention_head_path $result_path/identify_attention_head/attribution_result.json \
    --output-path $result_path/analyze_attention_inheritance \
    --temperature 0 \
    --conv-mode vicuna_v1

