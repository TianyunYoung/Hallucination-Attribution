export HF_HUB_OFFLINE=True

SEED=42
DATE=`date "+%m%d"`

model_name=llava-1.5 # or minigpt4
model_name=minigpt4
max_new_tokens=128
num_samples=50
data_path=../dataset
method=tfhh

## coco dataset
dataset=coco
if [ $model_name == 'llava-1.5' ]; then
    model_path=../checkpoints/llava-v1.5-7b-targeted-finetuned
elif [ $model_name == 'minigpt4' ]; then
    model_path=../checkpoints/minigpt4_checkpoint.pth
fi

EXP=${model_name}_${method}_token${max_new_tokens}_${SEED}_n${num_samples}
result_path=./results/$dataset/$DATE/$EXP
echo $result_path

CUDA_VISIBLE_DEVICES='0' python -m eval_scripts.eval_caption \
--model $model_name \
--dataset_name $dataset \
--image_folder $data_path/coco/val2014 \
--caption_file_path $data_path/coco/annotations/captions_val2014.json \
--seed $SEED \
--max_new_tokens $max_new_tokens \
--num_samples $num_samples \
--output_dir $result_path \
--decoder greedy \
--merged_ckpt $model_path

python eval_scripts/eval_utils/eval_chair.py \
    --annotation-dir $data_path/coco/annotations \
    --answers-file $result_path/captions.jsonl \
    --caption_file captions_val2014.json


# ## nocaps dataset
# dataset=nocaps
# EXP=${model_name}_${method}_token${max_new_tokens}_${SEED}_n${num_samples}
# result_path=./results/$dataset/$DATE/$EXP
# echo $result_path

# CUDA_VISIBLE_DEVICES=$GPU_ID python -m eval_scripts.eval_caption \
# --model $model_name \
# --dataset_name $dataset \
# --image_folder $data_path/nocaps/images/val \
# --caption_file_path $data_path/nocaps/annotations/nocaps_val_4500_captions.json \
# --seed $SEED \
# --max_new_tokens $max_new_tokens \
# --num_samples $num_samples \
# --output_dir $result_path \
# --decoder greedy \
# --merged_ckpt $model_path

# python eval_scripts/eval_utils/eval_nocaps_chair.py \
# --annotation-dir $data_path/nocaps/images/val \
# --answers-file $result_path/captions.jsonl \
# --caption_file nocaps_val_4500_captions.json
