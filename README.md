# Enhancing Parameter Efficiency and Generalization in Large Models: A Regularized and Masked Low-Rank Adaptation Approach

This repository contains the implementation of multiple LoRA variants from the paper titled **“Enhancing Parameter Efficiency and Generalization in Large Models: A Regularized and Masked Low-Rank Adaptation Approach”**. 


## Setup Environment

### Create a new conda environment:
```
conda env create -f env.yml
conda activate py39
```

## Adapt LoRA Variants on GLUE Benchmark

### This example runs standard LoRA on MRPC task

```
#!/bin/bash

export TASK_NAME=mrpc

CUDA_VISIBLE_DEVICES=0,1,2,3 python AdaLoRA/run_glue.py \
--model_name_or_path /microsoft/deberta-v3-base \
--dataset_name /huggingface/datasets/glue/$TASK_NAME \
--apply_lora \
--lora_type frd \
--lora_r 16  \
--reg_orth_coef 0 \
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
--output_dir output/deberta_lora_lr1e03_r16_coef0_alpha32/$TASK_NAME \
--overwrite_output_dir
```

#### Adajustable Hyperparameter 

+ `lora_r`: The rank of each LoRA incremental matrix. 
+ `lora_alpha`: The scaling of LoRA increments.
+ `learning_rate`: The learning rate.
+ `num_train_epochs`: Total number of training epochs.


### This example runs AdaLoRA on MRPC task

```
#!/bin/bash

export TASK_NAME=mrpc

CUDA_VISIBLE_DEVICES=0,1,2,3 python AdaLoRA/run_glue.py \
--model_name_or_path /microsoft/deberta-v3-base \
--dataset_name /huggingface/datasets/glue/$TASK_NAME \
--apply_lora \
--apply_adalora \
--lora_type svd \
--target_rank 8  \
--lora_r 16  \
--reg_orth_coef 0.1 \
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
--output_dir output/deberta_adalora_lr1e03_r16_8_coef01_alpha32/$TASK_NAME \
--overwrite_output_dir
```

#### Hyperparameter Setup

+ `lora_r`: The initial rank of each incremental matrix. 
+ `lora_alpha`: The scaling of LoRA increments.
+ `target_rank`: The average target rank of final incremental matrices, i.e. the average number of singular values per matrix. 
+ `init_warmup`: The steps of initial warmup for budget scheduler.
+ `final_warmup`: The steps of final warmup for budget scheduler.
+ `reg_orth_coef`: The weight of orthongonal regularization (0, 0.1).

### This example runs SoRA on MRPC task

```
#!/bin/bash

export TASK_NAME=mrpc

CUDA_VISIBLE_DEVICES=0,1,2,3 python SoRA/run_glue.py \
--model_name_or_path /microsoft/deberta-v3-base \
--dataset_name /huggingface/datasets/glue/$TASK_NAME \
--train_sparse True \
--sparse_lambda 0.000 \
--sparse_lambda_2 0.005 \
--apply_lora True \
--lora_r 16  \
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
--output_dir output/deberta_sora_lr1e03_r16_lambda_0_lambda2_0005_alpha32/$TASK_NAME \
--overwrite_output_dir
```

#### Adajustable Hyperparameter 

+ `lora_r`: The rank of each LoRA incremental matrix. 
+ `lora_alpha`: The scaling of LoRA increments.
+ `lora_alpha`: The scaling of LoRA increments.
+ `sparse_lambda_2`: The sparsity controller(0.1, 0.01, 0.001, 0.0001)



### This example runs RM-LoRA on MRPC task

```
#!/bin/bash

export TASK_NAME=mrpc

CUDA_VISIBLE_DEVICES=0,1,2,3 python RM-LoRA/run_glue.py \
--model_name_or_path /microsoft/deberta-v3-base \
--dataset_name /huggingface/datasets/glue/$TASK_NAME \
--apply_lora \
--lora_r 4  \
--random_rank 2 \
--reg_orth_coef 0.1 \
--lora_type v1 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 16 \
--do_train \
--do_eval \
--per_device_train_batch_size 32 \
--learning_rate 1e-3 \
--num_train_epochs 5 \
--max_seq_length 128 \
--report_to tensorboard \
--evaluation_strategy epoch \
--logging_step 10 \
--output_dir output/deberta_randomlora_lr1e03_r4_2_coef01_svd_alpha16/$TASK_NAME \
--overwrite_output_dir
```

#### Adajustable Hyperparameter 

+ `lora_r`: The rank of each LoRA incremental matrix. 
+ `random_rank`: The number of randomly updated directions of LoRA in each epoch.
+ `lora_type`: Choose from v1 and v2 to enable different variants of Random LoRA.
+ `lora_alpha`: The scaling of LoRA increments.
+ `reg_orth_coef`: The weight of orthongonal regularization (0, 0.1).
