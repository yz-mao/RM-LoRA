#!/bin/bash

export TASK_NAME=mrpc

CUDA_VISIBLE_DEVICES=2,3 python run_glue.py \
--model_name_or_path /home/zihao/PycharmProjects/Vit_Lora/microsoft/deberta-v3-base \
--dataset_name /home/zihao/PycharmProjects/Vit_Lora/AdaLoRA/Text_Classification/huggingface/datasets/glue/$TASK_NAME \
--apply_lora \
--apply_adalora True \
--lora_type svd \
--target_rank 8  \
--lora_r 16  \
--reg_orth_coef 0 \
--init_warmup 50 \
--final_warmup 50 \
--mask_interval 10 \
--beta1 0.85 \
--beta2 0.85 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 32 \
--do_train \
--do_eval \
--per_device_train_batch_size 32 \
--learning_rate 1e-3 \
--num_train_epochs 5 \
--max_seq_length 128 \
--report_to tensorboard \
--evaluation_strategy epoch \
--logging_step 10 \
--output_dir output/deberta_adalora_lr1e03_r16_8_coef0_alpha32/$TASK_NAME \
--overwrite_output_dir