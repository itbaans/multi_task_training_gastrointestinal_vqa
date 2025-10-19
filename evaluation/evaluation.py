import subprocess
subprocess.check_call(["pip", "install", "evaluate"])
subprocess.check_call(["pip", "install", "rouge_score"])

import requests
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import os
import json
import gc
from tqdm.notebook import tqdm
import torch
from peft import LoraConfig, get_peft_model, PeftModel

from transformers import AutoProcessor, AutoModelForCausalLM 

from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from datasets import load_dataset, Image as HfImage

import cv2

from evaluate import load

# Load metrics
bleu = load("bleu")
rouge = load("rouge")
meteor = load("meteor")

# from huggingface_hub import login

# login(token="api_token_here")

# IMG_DIR = "data/images"

torch_dtype = torch.float32

try:
    model = AutoModelForCausalLM.from_pretrained("your/path/to/model", torch_dtype=torch_dtype, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("your/path/to/model", trust_remote_code=True)
except Exception as e:
    print("Saved model not found initialzing new")
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=torch_dtype, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)


model = PeftModel.from_pretrained(
    model,
    "peeache/FL2_VQA_MIX_128_256",
    torch_dtype=torch_dtype,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


import cv2
import numpy as np
import matplotlib.pyplot as plt
import re

def visualize_model_output(raw_string, image):
    # Extract location values
    loc_values = [int(v) for v in re.findall(r"<loc_(\d+)>", raw_string)]

    # Group into (x, y) coordinate pairs
    points = list(zip(loc_values[::2], loc_values[1::2]))

    # Convert to NumPy array for OpenCV
    pts = np.array(points, dtype=np.int32)

    # Draw polygon on a copy of the image
    overlay = image.copy()
    cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Display the result
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Model Output Overlay")
    plt.show()

    return overlay

colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

from PIL import Image, ImageDraw, ImageFont 
import random

def draw_polygons(image, prediction, fill_mask=True):

    draw = ImageDraw.Draw(image)
    scale = 1
    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        print(polygons)
        print(label)
        color = random.choice(colormap)
        fill_color = random.choice(colormap) if fill_mask else None
        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue
            _polygon = (_polygon * scale).reshape(-1).tolist()
            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)

    plt.imshow(np.array(image))
    plt.axis('off')
    plt.title("Model Output Overlay")
    plt.show()
    return image

from typing import Dict, Any
import matplotlib.patches as patches

def plot_bbox(image: Image.Image, data: Dict[str, Any]):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
    ax.axis('off')
    return fig


def create_mask(image, prediction):
    from PIL import Image, ImageDraw
    import numpy as np
    import matplotlib.pyplot as plt

    mask = Image.new('L', image.size, 0)  
    draw = ImageDraw.Draw(mask)

    scale = 1
    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue
            
            _polygon = (_polygon * scale).reshape(-1).tolist()
            draw.polygon(_polygon, outline=255, fill=255)

            try:
                draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=255)
            except:
                pass

    # For visualization
    # plt.imshow(np.array(mask), cmap='gray')
    # plt.axis('off')
    # plt.title("Black and White Mask")
    # plt.show()

    # Return as numpy array (0 and 255)
    return np.array(mask, dtype=np.uint8)

def apply_mask_and_save(image, mask, output_path, alpha=0.3, mask_color=(255, 0, 0)):
    """
    Blend a semi-transparent colored mask onto the image.
    """
    # Ensure binary mask (0 or 1)
    mask_bool = (mask > 128).astype(np.uint8)

    masked_image = image.copy()
    overlay = np.full_like(image, mask_color, dtype=np.uint8)

    # Precompute blend
    blend = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    # Apply only where mask == 1
    masked_image[mask_bool == 1] = blend[mask_bool == 1]

    cv2.imwrite(output_path, masked_image)


def get_IoU(mask1, mask2):
    """
    Calculate Intersection over Union (IoU) between two masks.
    """
    # Convert to numpy arrays
    mask1 = np.array(mask1)
    mask2 = np.array(mask2)

    # Resize mask2 to match mask1 if sizes differ
    if mask1.shape[:2] != mask2.shape[:2]:
        #print("Warninig sizzes not same resizing")
        mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Convert RGB to grayscale if needed
    if len(mask1.shape) == 3:
        mask1 = cv2.cvtColor(mask1, cv2.COLOR_RGB2GRAY)
    if len(mask2.shape) == 3:
        mask2 = cv2.cvtColor(mask2, cv2.COLOR_RGB2GRAY)

    # Convert to binary arrays
    mask1 = mask1 > 128
    mask2 = mask2 > 128

    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    # Avoid division by zero
    if union == 0:
        return 0.0

    return intersection / union

def get_llm_judged_score(prediction, reference):
    """
    Placeholder for LLM-as-judge scoring function.
    To be implemented later.
    """
    # TODO: Implement LLM-based evaluation
    return 0.0

def get_segmentation_scores(data, model, MASK_DIR):
    
    predictions = []
    
    # --- main loop ---
    for img_id, mask_id, phrase in data:
    
        # Load full image
        image = cv2.imread(f"{IMG_DIR}/{img_id}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    
        # ---------------------------
        # Step 1: VQA inference
        prompt = f"<REFERRING_EXPRESSION_SEGMENTATION> {phrase}"
        inputs = processor(text=prompt, images=[image],
                           return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
            )
    
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        seg_answer = processor.post_process_generation(
            generated_text, task="<REFERRING_EXPRESSION_SEGMENTATION>",
            image_size=(image.width, image.height)
        )['<REFERRING_EXPRESSION_SEGMENTATION>']

        pred_mask = create_mask(image, seg_answer)
        score=0
        try:
            act_mask_path = f"{MASK_DIR}/{mask_id}.jpg"
            act_mask = Image.open(act_mask_path)
            score = get_IoU(act_mask, pred_mask)
        except FileNotFoundError:
            #print(f"Warning: Mask file not found at {act_mask_path}. Skipping score update.")
            continue

        predictions.append(score)

    return np.mean(predictions)

def get_vqa_exp_scores(img_ques, model):

    predictions = []
    
    # --- main loop ---
    for img_id, question, act_answer in img_ques:
    
        # Load full image
        image = cv2.imread(f"{IMG_DIR}/{img_id}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    
        # ---------------------------
        # Step 1: VQA inference
        prompt = f"<MedVQA_EXPLAIN> {question} Explain in Detail."
        inputs = processor(text=prompt, images=[image],
                           return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
            )
    
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        vqa_answer_exp = processor.post_process_generation(
            generated_text, task="<MedVQA_EXPLAIN>",
            image_size=(image.width, image.height)
        )['<MedVQA_EXPLAIN>']

        score = get_llm_judged_score(vqa_answer_exp, act_answer)
        predictions.append(score)

    return np.mean(predictions)

def get_vqa_scores(img_ques, model):
    
    predictions = []
    references = []
    
    # --- main loop ---
    for img_id, question, act_answer in img_ques:
    
        # Load full image
        image = cv2.imread(f"{IMG_DIR}/{img_id}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    
        # ---------------------------
        # Step 1: VQA inference
        prompt = f"<MedVQA> {question}"
        inputs = processor(text=prompt, images=[image],
                           return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
            )
    
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        vqa_answer = processor.post_process_generation(
            generated_text, task="<MedVQA>",
            image_size=(image.width, image.height)
        )['<MedVQA>']
    
        predictions.append(vqa_answer)
        references.append(act_answer)
    
    bleu_result = bleu.compute(predictions=predictions, references=references)
    rouge_result = rouge.compute(predictions=predictions, references=references)
    meteor_result = meteor.compute(predictions=predictions, references=references)
    
    scores = {
        'bleu': round(bleu_result['bleu'], 4),
        'rouge1': round(float(rouge_result['rouge1']), 4),
        'rouge2': round(float(rouge_result['rouge2']), 4),
        'rougeL': round(float(rouge_result['rougeL']), 4),
        'meteor': round(float(meteor_result['meteor']), 4)
    }

    return scores

def run_evaluation(task, model, model_id, test):
    if task == 'vqa':
        df = pd.read_parquet("data/splitted/train/vqa_combined.parquet")
        test_data = list(zip(
            df["img_id"],
            df["question"],
            df["answer"]
        ))
        if(test) : test_data=test_data[:10]
        score = {'model_id': model_id, 'scores': get_vqa_scores(test_data, model), 'task':task}
        with open(f"{model_id}_vqa_scores.json", 'w') as f:
            json.dump(score, f, indent=4)
            
    elif task == 'vqa_exp':
        df = pd.read_csv("data/splitted/test/vqa_exp.csv")
        test_data = list(zip(
            df["img_id"],
            df["question"],
            df["exp_ans"]
        ))
        if(test) : test_data=test_data[:10]
        score = {'model_id': model_id, 'score': get_vqa_exp_scores(test_data, model), 'task':task}
        with open(f"{model_id}_vqa_exp_scores.json", 'w') as f:
            json.dump(score, f, indent=4)
            
    elif task == 'seg_pseudo':
        df1 = pd.read_csv("data/splitted/test/cecum_mask_phrases.csv")
        df2 = pd.read_csv("data/splitted/test/oesophatigis_mask_phrases.csv")
        df3 = pd.read_csv("data\splitted\test\ulcerative_colitis_mask_phrases.csv")
        df4 = pd.read_csv("data\splitted\test\z-line_mask_phrases.csv")
        df = pd.concat([df1, df2, df3, df4], ignore_index=True)
        test_data = list(zip(
            df["img_id"],
            df["mask_id"],
            df["answer"]
        ))
        if(test) : test_data=test_data[:10]
        score = {'model_id': model_id, 'score': get_segmentation_scores(test_data, model, MASK_DIR="data/pseudo_masks"), 'task':task}
        with open(f"{model_id}_pseudo_seg_scores.json", 'w') as f:
            json.dump(score, f, indent=4)

    elif task == 'seg_polyp':
        df = pd.read_csv("data\splitted\test\polyps_mask_phrases.csv")
        test_data = list(zip(
            df["img_id"],
            df["mask_id"],
            df["answer"]
        ))
        if(test) : test_data=test_data[:10]
        score = {'model_id': model_id, 'score': get_segmentation_scores(test_data, model, MASK_DIR="data/polyp_masks"), 'task':task}
        with open(f"{model_id}_polyp_seg_scores.json", 'w') as f:
            json.dump(score, f, indent=4)

    elif task == 'seg_instrument':
        df = pd.read_csv("data\splitted\test\instruments_mask_phrases.csv")
        test_data = list(zip(
            df["img_id"],
            df["mask_id"],
            df["answer"]
        ))
        if(test) : test_data=test_data[:10]
        score = {'model_id': model_id, 'score': get_segmentation_scores(test_data, model, MASK_DIR="data/instruments_masks"), 'task':task}
        with open(f"{model_id}_instrument_seg_scores.json", 'w') as f:
            json.dump(score, f, indent=4)

## MIX is for Multi-Task trained model
run_evaluation("vqa", model, "FL2_VQA_MIX_128_256", False)
run_evaluation("seg_pseudo", model, "FL2_VQA_MIX_128_256", False)
run_evaluation("seg_instrument", model, "FL2_VQA_MIX_128_256", False)
run_evaluation("seg_polyp", model, "FL2_VQA_MIX_128_256", False)
