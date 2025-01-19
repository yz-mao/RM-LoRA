#!/bin/bash

export TASK_NAME=mrpc

CUDA_VISIBLE_DEVICES=0,1 python run_glue.py \
--model_name_or_path /home/zihao/PycharmProjects/Vit_Lora/microsoft/deberta-v3-base \
--dataset_name /home/zihao/PycharmProjects/Vit_Lora/AdaLoRA/Text_Classification/huggingface/datasets/glue/$TASK_NAME \
--apply_lora \
--lora_r 8  \
--random_rank 4 \
--reg_orth_coef 0.1 \
--lora_type v2 \
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
--output_dir output/deberta_randomlora_lr1e03_r8_4_coef01_v2_alpha32/$TASK_NAME \
--overwrite_output_dir