#!/bin/bash

set -e 
set -x

export HF_HOME="~/.cache/huggingface"

DATA_PATH="./dataset/playground/data"
LOG_PATH="./log"
SEED=1234
TIME_STEP=`date "+%Y-%m-%d-%H-%M-%S"`
OUTPUT="${LOG_PATH}/fine_tuning-llava_7b-$TIME_STEP-$SEED"
mkdir -p $OUTPUT

HEAD_PATH=./results/coco/llava_3000/identify_attention_head/attribution_result.json
PORT=$(printf "124%02d" $((RANDOM%100)))

deepspeed --master_port $PORT --num_gpus 1 \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path "${DATA_PATH}/train/llava_v1_5_mix665k.json" \
    --image_folder "${DATA_PATH}/train" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT \
    --seed $SEED \
    --num_train_epochs 1 \
    --max_steps 200 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --selective_tuning True \
    --random_selection False \
    --selective_tuning_top_k 30 \
    --fine_tuning_last_layer True \
    --attention_head_path $HEAD_PATH \
    --ce_loss_coeff 1.0 \
    --attention_loss_coeff 2.0 \
    --attention_loss "minimize_txt" \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
 2>&1 | tee $OUTPUT/training.log
