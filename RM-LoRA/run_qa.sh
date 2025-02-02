CUDA_VISIBLE_DEVICES=0,1 python run_qa.py \
--model_name_or_path /home/zihao/PycharmProjects/Vit_Lora/microsoft/deberta-v3-base \
--dataset_name /home/zihao/PycharmProjects/Vit_Lora/data/squad \
--apply_lora \
--lora_r 8  \
--random_rank 2 \
--reg_orth_coef 0 \
--lora_type svd \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 16 \
--do_train \
--do_eval \
--per_device_train_batch_size 16 \
--learning_rate 1e-3 \
--num_train_epochs 1 \
--max_seq_length 384 \
--doc_stride 128 \
--report_to tensorboard \
--evaluation_strategy epoch \
--logging_step 10 \
--output_dir output/deberta_randomlora_lr1e03_r8_2_coef0_svd_alpha16/squad/ \
--overwrite_output_dir