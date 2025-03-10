export HF_HUB_OFFLINE=True

model_name=llama-3.2-11b
model_path=meta-llama/Llama-3.2-11B-Vision-Instruct
# model_name=chameleon-7b
# model_path=facebook/chameleon-7b
# model_name=chameleon-30b
# model_path=facebook/chameleon-30b

dataset=coco
data_path=../datasets
top_k=10
num_samples=500

result_path=./results/$dataset/${model_name}_top${top_k}_${num_samples}

for i in {0..3}; do
    start_idx=$((125 * i))
    end_idx=$((125 * (i + 1)))
    echo $start_idx $end_idx
    answers_file="$result_path/captions_$i.jsonl"
    gpu_id=$i
    CUDA_VISIBLE_DEVICES=$gpu_id python -m eval_scripts.eval_caption \
        --model-path $model_path \
        --image-folder $data_path/coco/val2014 \
        --caption_file_path $data_path/coco/annotations/captions_val2014.json \
        --answers-file $answers_file \
        --dataset $dataset \
        --num_samples $num_samples \
        --top_k $top_k \
        --prune \
        --start_idx $start_idx \
        --end_idx $end_idx &
done

wait
echo "All processes are complete."
cat $result_path/captions_0.jsonl $result_path/captions_1.jsonl $result_path/captions_2.jsonl $result_path/captions_3.jsonl > $result_path/captions.jsonl
echo "merged done"
