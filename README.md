<!---
Copyright 2023 The UKP Lab. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# FUN with Fisher: Improving Generalization of Adapter-Based Cross-lingual Transfer with Scheduled Unfreezing

This project contains experiment code for the respective publication: training task adapters by progressively unfreezing them during the training. 
This is built on a Fork of the Hugging Face Transformers library and the adapter-transformer library.

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
If you encounter any issues, please do not hesitate to email us at:
chen.liu AT tu-darmstadt DOT de

## Installation
Please see the original Adapter_README.md for adapter-transformers.
From source by cloning the repository:

```
git clone 
cd naccl2024-fun
pip install -e .
```

## Getting Started
The scripts for running our experiments are:

1. ``examples/pytorch/question-answering/run_qa.py`` - XQuAD/MLQA
2. ``examples/pytorch/multiple-choice/run_copa.py`` - XCOPA
3. ``examples/pytorch/text-classification/run_xnli.py`` - XNLI

Note: Please comment out the `import wandb` if you don't wish to send results to wandb. 
Alternatively, to enable logging to wandb, you need to add something like this
``wandb.init(project="myproject", entity="abcde")
``
to the scripts.

Other than the args from the original adapter transformers, there are several important args:
1. ``--freeze_all`` [bool] This option will freeze all layers for "roberta" or "bert" based-model. This is 
required to use all scheduled unfreezing.
   
2. ``--use_gu`` [bool] This option allows you to run gradual unfreezing on predetermined intervals. 

3. ``--use_schedule`` [bool] This option allows you to turn on other scheduled unfreezing method. 

4. ``--unfreeze_interval`` [str] Unfreezing interval. Available ones are: `50-12`, `100-12`, `800-12`, `1000-12`. 
   
5. ``--schedule_style`` [str] This option allows you to choose from one of the schduled unfreezing: "lpft", "one", 
   "rand" or a pre-setted schedule (e.g. "schedule-1"). Note, you don't need this to run *gradual_unfreezing*.
   
6. ``--exp_name`` [str] Experiment name to report to wandb.



## To run an experiment
Assume you want to run on qa dataset, with gradual unfreezing:
```
python run_qa.py \
  --model_name_or_path xlm-roberta-base \
  --dataset_name squad \
  --seed $seed \
  --do_train \
  --do_eval \
  --freeze_all \ # feeze all the adapters initially
  --use_gu \ # use gradual unfreezing
  --train_adapter \
  --adapter_config "pfeiffer+inv" \
  --load_lang_adapter en/wiki@ukp \
  --language en \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 5e-4 \
  --num_train_epochs 15 \
  --overwrite_output_dir \
  --save_total_limit 2 \
  --load_best_model_at_end \
  --evaluation_strategy steps \
  --metric_for_best_model eval_f1 \
  --greater_is_better True \
  --max_seq_length 384 \
  --doc_stride 128 \
  --exp_name xlmr_squad_gu_$seed \
  --output_dir squad_gu_$seed
```

If you want to run on QA dataset, with FUN, unfreeze at every 50 steps:

```
python run_qa.py \
  --model_name_or_path xlm-roberta-base \
  --dataset_name squad \
  --seed $seed \
  --do_train \
  --do_eval \
  --freeze_all \   # feeze all the adapters initially
  --use_schedule \  # use a schedule other than gu
  --schedule_style one \  # unfreeze one adapter at a time using tr(F)
  --unfreeze_interval 50-12 \  # unfreeze an adapter every 50 steps
  --train_adapter \
  --adapter_config "pfeiffer+inv" \
  --load_lang_adapter en/wiki@ukp \
  --language en \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 5e-4 \
  --num_train_epochs 15 \
  --overwrite_output_dir \
  --save_total_limit 2 \
  --load_best_model_at_end \
  --evaluation_strategy steps \
  --metric_for_best_model eval_f1 \
  --greater_is_better True \
  --max_seq_length 384 \
  --doc_stride 128 \
  --exp_name xlmr_squad_fun_$seed \
  --output_dir squad_fun_$seed
```

## Citation

If you use this for your work, please consider citing our paper, as well as the AdapterHub.
```
@inproceedings{liu-etal-2024-fun,
    title = "{FUN} with Fisher: Improving Generalization of Adapter-Based Cross-lingual Transfer with Scheduled Unfreezing",
    author = "Liu, Chen  and
      Pfeiffer, Jonas  and
      Vuli{\'c}, Ivan  and
      Gurevych, Iryna",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.111/",
    doi = "10.18653/v1/2024.naacl-long.111",
    pages = "1998--2015"}
```
