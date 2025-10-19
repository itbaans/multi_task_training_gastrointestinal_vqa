import torch
import requests
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import os
import json
import gc
from tqdm.notebook import tqdm

from transformers import AutoProcessor, AutoModelForCausalLM 

from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from datasets import load_dataset, Image as HfImage

import cv2

from huggingface_hub import login

login(token="api_here")

import torch
from peft import LoraConfig, get_peft_model, PeftModel

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
    "peeache/Florence-2-Final-Batched-PipeLine-LoRA-128-256",
    torch_dtype=torch_dtype,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

import pandas as pd
from datasets import load_dataset, Image as HfImage

ds = load_dataset("SimulaMet/Kvasir-VQA-x1")["test"]
val_set_task2 = (
    ds.filter(lambda x: x["complexity"] == 1)
      .shuffle(seed=42)
      .select(range(1500))
      .add_column("val_id", list(range(1500)))
      .remove_columns(["complexity", "answer", "original", "question_class"])
      .cast_column("image", HfImage())
)

print(f"Final size after filtering: {len(val_set_task2)}")

IMG_DIR = "data/images"

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
    plt.imshow(np.array(mask), cmap='gray')
    plt.axis('off')
    plt.title("Black and White Mask")
    plt.show()

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

def conf_score(model, processor, image, question, device, task, 
                                      special_text="", max_new_tokens=128, k=5):
    prompt = f"{task} {question} {special_text}"
    inputs = processor(text=prompt, images=[image],
                       return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Florence2 needs decoder_input_ids to start generation
    decoder_input_ids = torch.tensor(
        [[model.generation_config.decoder_start_token_id]],
        device=device
    )

    all_token_info = []
    stability_scores = []  # per-step top-k masses

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],        # encoder input
                pixel_values=inputs["pixel_values"],  # multimodal
                decoder_input_ids=decoder_input_ids   # growing decoder side
            )
            logits = outputs.logits[:, -1, :]  # last step logits
            probs = F.softmax(logits, dim=-1)

            # ---- TOP-K STABILITY MASS ----
            topk_probs, _ = torch.topk(probs, k, dim=-1)
            topk_mass = topk_probs.sum().item()
            stability_scores.append(topk_mass)

            # Greedy decode
            next_id = torch.argmax(probs, dim=-1)
            next_prob = probs[0, next_id].item()

        token_str = processor.tokenizer.decode([next_id.item()], skip_special_tokens=True)
        all_token_info.append({
            "token_id": next_id.item(),
            "token_str": token_str,
            "prob": next_prob,
            "topk_mass": topk_mass
        })

        # Append to decoder sequence
        decoder_input_ids = torch.cat([decoder_input_ids, next_id.unsqueeze(0)], dim=-1)

        # Stop if EOS
        if next_id.item() == processor.tokenizer.eos_token_id:
            break

    # ---- Final Top-k Stability Confidence ----
    topk_stability_conf = float(np.mean(stability_scores)) if stability_scores else 0.0

    return topk_stability_conf

import torch
import torch.nn.functional as F
import numpy as np

def explain_with_token_probs_and_conf(model, processor, image, question, device, 
                                      max_new_tokens=128, k=5):
    """
    Run <MedVQA_EXPLAIN> and capture tokens with their probabilities
    + compute top-k stability confidence.
    """
    prompt = f"<MedVQA_EXPLAIN> {question} Explain in Detail."
    inputs = processor(text=prompt, images=[image],
                       return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Florence2 needs decoder_input_ids to start generation
    decoder_input_ids = torch.tensor(
        [[model.generation_config.decoder_start_token_id]],
        device=device
    )

    all_token_info = []
    stability_scores = []  # per-step top-k masses

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],        # encoder input
                pixel_values=inputs["pixel_values"],  # multimodal
                decoder_input_ids=decoder_input_ids   # growing decoder side
            )
            logits = outputs.logits[:, -1, :]  # last step logits
            probs = F.softmax(logits, dim=-1)

            # ---- TOP-K STABILITY MASS ----
            topk_probs, _ = torch.topk(probs, k, dim=-1)
            topk_mass = topk_probs.sum().item()
            stability_scores.append(topk_mass)

            # Greedy decode
            next_id = torch.argmax(probs, dim=-1)
            next_prob = probs[0, next_id].item()

        token_str = processor.tokenizer.decode([next_id.item()], skip_special_tokens=True)
        all_token_info.append({
            "token_id": next_id.item(),
            "token_str": token_str,
            "prob": next_prob,
            "topk_mass": topk_mass
        })

        # Append to decoder sequence
        decoder_input_ids = torch.cat([decoder_input_ids, next_id.unsqueeze(0)], dim=-1)

        # Stop if EOS
        if next_id.item() == processor.tokenizer.eos_token_id:
            break

    # Decode the whole sequence
    generated_text = processor.batch_decode(decoder_input_ids, skip_special_tokens=False)[0]
    vqa_answer_exp = processor.post_process_generation(
        generated_text,
        task="<MedVQA_EXP>",
        image_size=(image.width, image.height)
    )['<MedVQA_EXP>']

    # ---- Final Top-k Stability Confidence ----
    topk_stability_conf = float(np.mean(stability_scores)) if stability_scores else 0.0

    return vqa_answer_exp, all_token_info, topk_stability_conf

import re
from nltk.corpus import stopwords

# Preload stopwords (English)
STOPWORDS = set(stopwords.words("english"))
STOPWORDS.add("image")
STOPWORDS.add("view")

def merge_tokens_with_probs(token_info_list):
    """
    Convert token-prob dicts to word-level with filtering and merging.
    
    Args:
        token_info_list: list of dicts [{"token_str": str, "prob": float, ...}, ...]

    Returns:
        List of (word, avg_prob) sorted from high -> low.
    """

    merged = []
    current_word = ""
    current_probs = []

    for tok in token_info_list:
        t = tok["token_str"]
        p = tok["prob"]

        # Skip empty, special tokens <...>
        if not t.strip() or re.match(r"^<.*>$", t):
            continue

        # Decide if this token starts a new word
        if t.startswith(" ") or current_word == "":
            # If we already have a word collected, push it
            if current_word:
                word = current_word.strip()
                if word.lower() not in STOPWORDS or word.lower() == "no":
                    merged.append((word, sum(current_probs)/len(current_probs)))
            # Start new word
            current_word = t
            current_probs = [p]
        else:
            # Continuation of previous word
            current_word += t
            current_probs.append(p)

        # If token ends with space, finalize immediately
        if t.endswith(" "):
            word = current_word.strip()
            if word.lower() not in STOPWORDS:
                merged.append((word, sum(current_probs)/len(current_probs)))
            current_word = ""
            current_probs = []

    # Flush leftover
    if current_word:
        word = current_word.strip()
        if word.lower() not in STOPWORDS:
            merged.append((word, sum(current_probs)/len(current_probs)))

    # Sort by probability (descending)
    #merged.sort(key=lambda x: x[1], reverse=True)

    return merged


# Pair img_id with question and ground truth answer
img_ques = list(zip(
    val_set_task2["val_id"],
    val_set_task2["img_id"],
    val_set_task2["question"],
))

img_ques = img_ques 

import json
from pathlib import Path
import time

vis_dir = "visuals"
os.makedirs(vis_dir, exist_ok=True)  # create folder if it doesn't exist
results = []  # will store all JSON records

# --- main loop ---
for val_id, img_id, question in img_ques:
    start_time = time.time()  # Start timing

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

    # Explanation + confidence
    vqa_answer_exp, token_probs, topk_stability_conf = explain_with_token_probs_and_conf(
        model, processor, image, question, device
    )

    # ---------------------------
    # Step 2: Segmentation inference
    inputs = processor(
        text=f"<REFERRING_EXPRESSION_SEGMENTATION> {vqa_answer}",
        images=[image],
        return_tensors="pt",
        padding=True
    )
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
        generated_text,
        task="<REFERRING_EXPRESSION_SEGMENTATION>",
        image_size=(image.width, image.height)
    )['<REFERRING_EXPRESSION_SEGMENTATION>']

    # ---------------------------
    # Step 3: Caption generation
    inputs = processor(
        text=f"<MORE_DETAILED_CAPTION>",
        images=[image],
        return_tensors="pt",
        padding=True
    )
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
    caption_answer = processor.post_process_generation(
        generated_text,
        task="<MORE_DETAILED_CAPTION>",
        image_size=(image.width, image.height)
    )['<MORE_DETAILED_CAPTION>']

    # ---------------------------
    # Step 4: Apply + save mask
    mask = create_mask(image, seg_answer)
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    vis_path = f"{vis_dir}/_mask_{val_id}.jpg"
    apply_mask_and_save(image_bgr, mask, vis_path)

    # ---------------------------
    # Step 5: Construct JSON record
    record = {
        "val_id": val_id,
        "img_id": img_id,
        "question": question,
        "answer": vqa_answer,
        "textual_explanation": f"{vqa_answer_exp}\nOverall explaination of image: {caption_answer}",
        "visual_explanation": [{
            "type": "segmentation_mask",
            "data": vis_path,
            "description": "Highlighted mask showing the region of interest supporting the answer."
        }],
        "confidence_score": float(topk_stability_conf)
    }

    results.append(record)
    elapsed_time = time.time() - start_time
    print(f"Processed val_id {val_id} in {elapsed_time:.2f} seconds")


# ---------------------------
# Save all results to JSON file
output_json = "submission_task2.jsonl"
with open(output_json, "w") as f:
    json.dump(results, f, indent=2)