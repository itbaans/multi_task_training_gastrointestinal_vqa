import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import numpy as np
import torch

from transformers import Trainer, TrainingArguments
import numpy as np

import wandb
from training.training_utils import (
    FlorenceCollator,
    KvasirVQADataset,
    load_florence_model,
    get_florence2_lora_targets,
    apply_lora_config,
    show_random_samples,
)

model_lora_configs = {

    "FL2_VQA_32_64": (32, 64),
    "FL2_VQA_64_128":(64, 128),
    "FL2_VQA_128_256":(128, 256),
    "FL2_VQA_MT_32_64": (32, 64),
    "FL2_VQA_MT_64_128":(64, 128),
    "FL2_VQA_MT_128_256":(128, 256),
    
}

model_id = "FL2_VQA_MT_128_256"
eval_training = False

wanb_key = "wandb_key_here"
wandb.login(key=wanb_key)

os.environ["WANDB_PROJECT"] = f"{model_id}"
os.environ["WANDB_DISABLED"] = "false"


model, processor = load_florence_model("microsoft/Florence-2-base", model_adapters=None)
target_modules, all_modules = get_florence2_lora_targets(model)

model = apply_lora_config(model, target_modules, rank=model_lora_configs[model_id][0], alpha=model_lora_configs[model_id][1])

IMG_DIR = "data/images"

dataset_dir = "data/splitted/train" if eval_training else "data/combined"

if eval_training:
    vqa_dataset = KvasirVQADataset(
        dataset=f"{dataset_dir}/vqa_combined.parquet",
        image_dir= IMG_DIR,
        task='<MedVQA>',
        processor=processor)
else:
    df1 = pd.read_parquet(f"{dataset_dir}/Kvasir-VQA-x1-train.parquet")
    df2 = pd.read_parquet(f"{dataset_dir}/Kvasir-VQA-x1-test.parquet")

    vqa_combined = pd.concat([df1, df2], ignore_index=True)
    vqa_dataset = KvasirVQADataset(
        dataset=vqa_combined,
        image_dir= IMG_DIR,
        task='<MedVQA>',
        processor=processor)
show_random_samples(vqa_dataset)

if "MT" not in model_id:
    training_dataset = vqa_dataset
else:
    instruments_dataset = KvasirVQADataset(
        dataset=f"{dataset_dir}/instruments_mask_phrases.csv",
        image_dir= IMG_DIR,
        mask_dir="data/instruments_masks",
        task='<REFERRING_EXPRESSION_SEGMENTATION>',
        tag='instrument_v2',
        processor=processor)

    polyp_dataset = KvasirVQADataset(
        dataset=f"{dataset_dir}/polyps_mask_phrases.csv",
        image_dir= IMG_DIR,
        mask_dir="data/polyp_masks",
        task='<REFERRING_EXPRESSION_SEGMENTATION>',
        tag='polyp',
        processor=processor)

    z_line_dataset = KvasirVQADataset(
        dataset=f"{dataset_dir}/z-line_mask_phrases.csv",
        image_dir= IMG_DIR,
        mask_dir="data/pseudo_masks",
        task='<REFERRING_EXPRESSION_SEGMENTATION>',
        tag='z-line',
        processor=processor)

    oesophagitis_dataset = KvasirVQADataset(
        dataset=f"{dataset_dir}/oesophatigis_mask_phrases.csv",
        image_dir= IMG_DIR,
        mask_dir="data/pseudo_masks",
        task='<REFERRING_EXPRESSION_SEGMENTATION>',
        tag='oesophagitis',
        processor=processor)

    ulcerative_colitis_dataset = KvasirVQADataset(
        dataset=f"{dataset_dir}/ulcerative_colitis_mask_phrases.csv",
        image_dir= IMG_DIR,
        mask_dir="data/pseudo_masks",
        task='<REFERRING_EXPRESSION_SEGMENTATION>',
        tag='ulcerative colitis',
        processor=processor)

    cecum_dataset = KvasirVQADataset(
        dataset=f"{dataset_dir}/cecum_mask_phrases.csv",
        image_dir= IMG_DIR,
        mask_dir="data/pseudo_masks",
        task='<REFERRING_EXPRESSION_SEGMENTATION>',
        tag='cecum',
        processor=processor)

    vqa_exp_dataset = KvasirVQADataset(
        dataset=f"{dataset_dir}/vqa_exp.csv",
        image_dir= IMG_DIR,
        task='<MedVQA_EXPLAIN>',
        processor=processor)


    show_random_samples(polyp_dataset)
    show_random_samples(z_line_dataset)
    show_random_samples(oesophagitis_dataset)
    show_random_samples(ulcerative_colitis_dataset)
    show_random_samples(cecum_dataset)
    show_random_samples(vqa_exp_dataset)
    show_random_samples(instruments_dataset)

    training_dataset = torch.utils.data.ConcatDataset([
        instruments_dataset,
        polyp_dataset,
        z_line_dataset,
        oesophagitis_dataset,
        ulcerative_colitis_dataset,
        cecum_dataset,
        vqa_exp_dataset,
        vqa_dataset
    ])

training_args = TrainingArguments(
    output_dir="./outputs",
    save_strategy="steps",        # save every N steps
    save_steps=100,               # adjust based on dataset size
    save_total_limit=1,           # keep only last 1 checkpoints
    logging_steps=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=3,
    num_train_epochs=1,
    learning_rate=5e-5,
    warmup_ratio=0.1,
    fp16=True,
    remove_unused_columns=False,
    report_to="wandb"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_dataset,
    data_collator=FlorenceCollator(processor),
    tokenizer=processor.tokenizer
)

trainer.train()

checkpoint_paths = []
for checkpoint in os.listdir(training_args.output_dir):
    if checkpoint.startswith("checkpoint-"):
        checkpoint_paths.append(os.path.join(training_args.output_dir, checkpoint))

model, processor = load_florence_model("microsoft/Florence-2-base", f"outputs/{checkpoint_paths[-1].split('/')[-1]}")

from huggingface_hub import login, whoami
HF_TOKEN = "API_HERE"

# Login programmatically (no CLI prompt needed)
login(token=HF_TOKEN)

HF_USER = whoami()["name"]
print("Logged into HF as:", HF_USER)

model.push_to_hub(f"username/{model_id}")
