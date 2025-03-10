export HF_HUB_OFFLINE=True

GPU_ID=0
SEED=42
DATE=`date "+%m%d"`

model_name=llava-1.5
max_new_tokens=128
num_samples=500
data_path=../dataset

# coco dataset
for method in greedy vcd dola halc opera
do
    EXP=${model_name}_${method}_token${max_new_tokens}_${SEED}_n${num_samples}
    result_path=./results/$dataset/$DATE/$EXP
    echo $result_path
    mkdir -p $result_path

    CUDA_VISIBLE_DEVICES=$GPU_ID python -m eval_scripts.eval_caption \
    --model $model_name \
    --dataset_name $dataset \
    --image_folder $data_path/coco/val2014 \
    --caption_file_path $data_path/coco/annotations/captions_val2014.json \
    --seed $SEED \
    --output_dir $result_path \
    --max_new_tokens $max_new_tokens \
    --num_samples $num_samples \
    --decoder $method \
    --verbosity

    python eval_scripts/eval_utils/eval_chair.py \
        --annotation-dir $data_path/coco/annotations \
        --answers-file $result_path/captions.jsonl \
        --caption_file captions_val2014.json

done

# nocaps dataset
for method in greedy vcd dola halc opera
do
    EXP=${model_name}_${method}_token${max_new_tokens}_${SEED}_n${num_samples}
    result_path=./results/$dataset/$DATE/$EXP
    echo $result_path
    mkdir -p $result_path

    CUDA_VISIBLE_DEVICES=$GPU_ID python -m eval_scripts.eval_caption \
    --model $model_name \
    --dataset_name $dataset \
    --image_folder $data_path/nocaps/images/val \
    --caption_file_path $data_path/nocaps/annotations/nocaps_val_4500_captions.json \
    --seed $SEED \
    --output_dir $result_path \
    --max_new_tokens $max_new_tokens \
    --num_samples $num_samples \
    --decoder $method \
    --verbosity

    python eval_scripts/eval_utils/eval_nocaps_chair.py \
    --annotation-dir $data_path/nocaps/images/val \
    --answers-file $result_path/captions.jsonl \
    --caption_file nocaps_val_4500_captions.json
done