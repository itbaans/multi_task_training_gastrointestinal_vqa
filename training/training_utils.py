import numpy as np
import pandas as pd

import os
import random
from typing import Tuple, List, Dict, Any

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import torch

from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel, LoraConfig, get_peft_model  # ✅ added
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import re

import traceback

# Global constants
coordinate_precision = 1000
colormap = ["red", "green", "blue", "yellow", "purple", "orange"]


def pixel_to_florence_coords(x: int, y: int, image_width: int, image_height: int) -> Tuple[int, int]:
    """Convert pixel coordinates to Florence-2 location token format"""
    norm_x = min(coordinate_precision, max(0, int((x / image_width) * coordinate_precision)))
    norm_y = min(coordinate_precision, max(0, int((y / image_height) * coordinate_precision)))
    return norm_x, norm_y


def polygon_to_florence_string(polygon, image_width, image_height) -> str:
    tokens = []
    for point in polygon:
        if point.ndim == 1:  # already [x, y]
            x, y = point
        else:  # OpenCV contour format [[x, y]]
            x, y = point[0]

        fx, fy = pixel_to_florence_coords(int(x), int(y), image_width, image_height)
        tokens.append(f"<loc_{fx}><loc_{fy}>")
    return "".join(tokens)

def bbox_to_florence_string(bbox: Tuple[int, int, int, int], image_width: int, image_height: int, tag: str) -> str:
    """Convert bounding box to Florence-2 location token string"""
    x_min, y_min, x_max, y_max = bbox
    # top-left
    x1, y1 = pixel_to_florence_coords(x_min, y_min, image_width, image_height)
    # bottom-right
    x2, y2 = pixel_to_florence_coords(x_max, y_max, image_width, image_height)

    return f"{tag}<loc_{x1}><loc_{y1}>{tag}<loc_{x2}><loc_{y2}>"


def approximate_contour(contour, epsilon_ratio=0.008):

    # First simplify using approxPolyDP
    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    
    return simplified

def mask_to_bboxes_florence_string(mask_img: Image.Image, tag: str) -> str:
    """Convert mask image to bounding boxes in Florence-2 string format"""
    mask_array = np.array(mask_img.convert("L"))  # ensure grayscale
    image_height, image_width = mask_array.shape

    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    florence_strings = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if tag and tag == 'polyp':
            if area < 1000:
                #print(area)
                continue
        else:
            if area < 35000:
                continue
        polygon = approximate_contour(contour)
        x, y, w, h = cv2.boundingRect(polygon)
        bbox = (x, y, x + w, y + h)
        florence_strings.append(bbox_to_florence_string(bbox, image_width, image_height, tag))

    return "".join(florence_strings) if florence_strings else ""


def mask_to_florence_string(mask_img: Image.Image, tag=None) -> str:
    mask_array = np.array(mask_img.convert("L"))  # grayscale
    binary_mask = (mask_array > 0).astype(np.uint8) * 255

    h, w = binary_mask.shape
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    florence_strings = []

    if tag == 'instrument':
        if not contours:
            return ""
        # pick contour with maximum area
        top_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        for contour in top_contours:
            polygon = approximate_contour(contour)
            florence_strings.append(polygon_to_florence_string(polygon, w, h))

    else:
        for contour in contours:
            area = cv2.contourArea(contour)
            if tag == 'polyp' and area < 1000:
                continue
            elif tag == 'instrument_v2' and area < 900:
                continue
            elif tag is None and area < 35000:
                continue

            polygon = approximate_contour(contour)
            florence_strings.append(polygon_to_florence_string(polygon, w, h))

    #print(len(florence_strings))
    return "".join(florence_strings) if florence_strings else ""


class KvasirVQADataset(Dataset):
    def __init__(self, dataset, processor, image_dir: str, task: str, mask_dir: str = None, tag: str = None):
        self.dataset = dataset
        self.processor = processor
        self.image_dir = image_dir
        self.task = task
        self.mask_dir = mask_dir
        self.tag = tag
        self.task_prefix_to_prompt = {
            '<MedVQA>': '<MedVQA> {input}',
            '<MedVQA_EXPLAIN>': '<MedVQA_EXPLAIN> {input}',
            '<REFERRING_EXPRESSION_SEGMENTATION>': '<REFERRING_EXPRESSION_SEGMENTATION> {input}',
            '<OD>': '<OD>'
        }

        # ---- PREPROCESSING STEP ----
        self.dataset = self._load_and_preprocess_dataset(dataset)


    def _load_and_preprocess_dataset(self, dataset):
        # Case 1: dataset is a CSV path
        if isinstance(dataset, str) and dataset.endswith(".csv"):
            df = pd.read_csv(dataset)
        # Case 2: dataset is a Parquet path
        elif isinstance(dataset, str) and dataset.endswith(".parquet"):
            df = pd.read_parquet(dataset)
        # Case 3: dataset is a pandas DataFrame
        elif isinstance(dataset, pd.DataFrame):
            df = dataset
        # Case 4: dataset is already a list of dicts
        elif isinstance(dataset, list):
            df = pd.DataFrame(dataset)
        else:
            raise ValueError("Unsupported dataset format. Provide CSV/Parquet path, DataFrame, or list of dicts.")

        # --- Pre-computation for Segmentation and OD tasks ---
        if self.task in ['<REFERRING_EXPRESSION_SEGMENTATION>', '<OD>'] and self.mask_dir:
            new_answers = []
            valid_indices = []
            
            for index, sample in df.iterrows():
                image_id = sample.get('img_id')
                mask_id = sample.get('mask_id')

                if image_id is None or mask_id is None:
                    continue

                image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
                if self.tag == 'instrument':
                    mask_path = os.path.join(self.mask_dir, f"{mask_id}")
                else:
                    mask_path = os.path.join(self.mask_dir, f"{mask_id}.jpg")
                if not os.path.exists(image_path) or not os.path.exists(mask_path):
                    continue

                try:
                    with Image.open(mask_path) as mask:
                        if np.all(np.array(mask) == 0):
                            continue
                        
                        # Pre-compute the answer string
                        if self.task == '<REFERRING_EXPRESSION_SEGMENTATION>':
                            answer = mask_to_florence_string(mask, tag=self.tag if self.tag in ['polyp', 'instrument', 'instrument_v2'] else None)
                        elif self.task == '<OD>':
                            answer = mask_to_bboxes_florence_string(mask, tag=self.tag)
                        else:
                            answer = "" # Should not happen based on the outer if
                        
                        new_answers.append(answer)
                        valid_indices.append(index)

                except Exception as e:
                    print(f"Skipping sample {index} due to error: {e}")
                    continue
            
            # Filter the dataframe to only include valid samples and add the new 'answer' column
            df = df.loc[valid_indices].copy()
            df['coord_string'] = new_answers
            
            print(f"[Preprocessing] Processed {len(df)} samples for task '{self.task}'.")

        return df.to_dict(orient="records")
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if isinstance(idx, slice):
            # return a new dataset object with a subset
            new_data = self.dataset[idx]
            return KvasirVQADataset(
                dataset=new_data,
                processor=self.processor,
                image_dir=self.image_dir,
                task=self.task,
                mask_dir=self.mask_dir,
                tag=self.tag
            )
        try:
            sample = self.dataset[idx]
            image_id = sample['img_id']
            image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
            image = Image.open(image_path).convert("RGB")

            if self.task == '<MedVQA>':
                task_prompt = self.task_prefix_to_prompt[self.task]
                question = sample['question']
                answer = sample['answer']
                prompt = task_prompt.format(input=question)
                return {'prompt': prompt, 'answer': answer, 'image': image}

            elif self.task == '<MedVQA_EXPLAIN>':
                task_prompt = self.task_prefix_to_prompt[self.task]
                question = sample['question']

                if "no" in sample['n_answer'].lower():
                    answer = sample['exp_ans'].strip().split('. ')[0] + '.'
                else:
                    answer = sample['exp_ans']
                
                text = f"{question} Explain in Detail."
                prompt = task_prompt.format(input=text)
                return {'prompt': prompt, 'answer': answer, 'image': image}

            elif self.task == '<REFERRING_EXPRESSION_SEGMENTATION>' and self.tag:
                task_prompt = self.task_prefix_to_prompt[self.task]
                phrase = sample['answer']
                # Directly fetch the pre-computed answer
                answer = sample['coord_string']
                prompt = task_prompt.format(input=phrase)
                return {'prompt': prompt, 'answer': answer, 'image': image}

            elif self.task == '<OD>' and self.tag:
                task_prompt = self.task_prefix_to_prompt[self.task]
                # Directly fetch the pre-computed answer
                answer = sample['coord_string']
                return {'prompt': task_prompt, 'answer': answer, 'image': image}

        except Exception as e:
            print(f"❌ Error processing item {idx}: {repr(e)}")
            traceback.print_exc()
            return {
                'prompt': "<MedVQA>What is this?",
                'answer': "Unknown",
                'image': Image.new('RGB', (224, 224), color='white')
            }


def collate_fn(batch: List[Dict[str, Any]], processor) -> Dict[str, torch.Tensor]:
    """Custom collate function to handle variable length sequences"""
    prompts = [item['prompt'] for item in batch]
    answers = [item['answer'] for item in batch]
    images = [item['image'] for item in batch]

    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    labels = processor.tokenizer(
        text=answers,
        return_tensors="pt",
        padding=True
    )

    # Replace padding token id's with -100 to ignore in loss
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    
    inputs['labels'] = labels['input_ids']
    
    return inputs

def load_florence_model(model_id: str, model_adapters: str = None):
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    if model_adapters:
        model = PeftModel.from_pretrained(model, model_adapters)
    return model, processor

def get_florence2_lora_targets(model):
    """Get appropriate LoRA target modules for Florence-2 model"""
    
    # Define patterns for modules we want to include
    target_patterns = [
        # Vision tower attention layers
        'vision_tower.blocks.*.*.window_attn.fn.qkv',
        'vision_tower.blocks.*.*.window_attn.fn.proj',
        'vision_tower.blocks.*.*.channel_attn.fn.qkv', 
        'vision_tower.blocks.*.*.channel_attn.fn.proj',
        
        # Vision tower MLP layers
        'vision_tower.blocks.*.*.ffn.fn.net.fc1',
        'vision_tower.blocks.*.*.ffn.fn.net.fc2',
        
        # Language model encoder attention
        'language_model.model.encoder.layers.*.self_attn.q_proj',
        'language_model.model.encoder.layers.*.self_attn.k_proj',
        'language_model.model.encoder.layers.*.self_attn.v_proj',
        'language_model.model.encoder.layers.*.self_attn.out_proj',
        
        # Language model encoder MLP
        'language_model.model.encoder.layers.*.fc1',
        'language_model.model.encoder.layers.*.fc2',
        
        # Language model decoder attention
        'language_model.model.decoder.layers.*.self_attn.q_proj',
        'language_model.model.decoder.layers.*.self_attn.k_proj', 
        'language_model.model.decoder.layers.*.self_attn.v_proj',
        'language_model.model.decoder.layers.*.self_attn.out_proj',
        
        # Language model decoder cross-attention
        'language_model.model.decoder.layers.*.encoder_attn.q_proj',
        'language_model.model.decoder.layers.*.encoder_attn.k_proj',
        'language_model.model.decoder.layers.*.encoder_attn.v_proj', 
        'language_model.model.decoder.layers.*.encoder_attn.out_proj',
        
        # Language model decoder MLP
        'language_model.model.decoder.layers.*.fc1',
        'language_model.model.decoder.layers.*.fc2',
    ]
    
    # Get all Linear layer names from the model
    all_linear_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            all_linear_modules.append(name)
    
    print(f"Found {len(all_linear_modules)} Linear modules total")
    
    # Modules to explicitly exclude (embeddings, heads, projections)
    exclude_patterns = [
        'embed_tokens',           # Token embeddings
        'embed_positions',        # Position embeddings  
        'shared',                 # Shared embeddings
        'lm_head',               # Language model head
        'row_embeddings',        # Image position embeddings
        'column_embeddings',     # Image position embeddings
        'image_proj_norm',       # Image projection normalization
    ]
    
    # Filter modules based on patterns
    target_modules = []
    
    for module_name in all_linear_modules:
        # Check if module should be excluded
        should_exclude = any(exclude_pattern in module_name for exclude_pattern in exclude_patterns)
        
        if not should_exclude:
            # Check if it matches our target patterns (attention and MLP layers)
            is_target = any(
                pattern_part in module_name 
                for pattern_part in ['attn', 'fc1', 'fc2', 'proj', 'qkv']
            )
            
            if is_target:
                target_modules.append(module_name)
    
    return target_modules, all_linear_modules


def apply_lora_config(model, target_modules, rank, alpha):
    # Create LoRA configuration
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,             # LoRA scaling parameter
        target_modules=target_modules,
        lora_dropout=0.05,           # LoRA dropout
        bias="none",                 # Don't adapt bias parameters
        task_type="FEATURE_EXTRACTION",  # Task type for Florence-2
    )
    print(f"\nApplying LoRA to model...")
    print(f"LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}, dropout={lora_config.lora_dropout}")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

class FlorenceCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        return collate_fn(batch, self.processor)
    


# Assuming coordinate_precision is 1000 (standard for Florence-2)
coordinate_precision = 1000

def florence_coords_to_pixel(norm_x: int, norm_y: int, image_width: int, image_height: int):
    """Convert Florence-2 location tokens back to pixel coordinates"""
    x = int((norm_x / coordinate_precision) * image_width)
    y = int((norm_y / coordinate_precision) * image_height)
    return x, y

def decode_florence_string_to_polygon(coord_string: str, image_width: int, image_height: int):
    """Decode Florence-2 coordinate string back to polygon points"""
    # Extract all coordinate tokens
    pattern = r'<loc_(\d+)><loc_(\d+)>'
    matches = re.findall(pattern, coord_string)
    
    polygon = []
    for norm_x_str, norm_y_str in matches:
        norm_x, norm_y = int(norm_x_str), int(norm_y_str)
        x, y = florence_coords_to_pixel(norm_x, norm_y, image_width, image_height)
        polygon.append((x, y))
    
    return polygon

def create_mask_from_polygon(polygon, image_width, image_height):
    """Create a binary mask from polygon coordinates"""
    mask = Image.new('L', (image_width, image_height), 0)
    if polygon:
        ImageDraw.Draw(mask).polygon(polygon, fill=255)
    return mask

def show_random_samples(dataset, n_samples=1):
    """Display n random samples from the dataset"""
    if len(dataset) < n_samples:
        n_samples = len(dataset)
    
    # Get random indices
    indices = random.sample(range(len(dataset)), n_samples)
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        
        print(f"\n=== Sample {i+1} (Index: {idx}) ===")
        print(f"Task: {dataset.task}")
        print(f"Prompt: {sample['prompt']}")
        print(f"Answer: {sample['answer']}")
        
        # For segmentation tasks, show original and masked images
        if dataset.task == '<REFERRING_EXPRESSION_SEGMENTATION>' and 'coord_string' in dataset.dataset[idx]:
            image = sample['image']
            coord_string = sample['answer']  # This is the coord_string
            
            # Decode coordinates to create mask
            polygon = decode_florence_string_to_polygon(coord_string, image.width, image.height)
            mask = create_mask_from_polygon(polygon, image.width, image.height)
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Mask
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title('Decoded Mask')
            axes[1].axis('off')
            
            # Overlay
            overlay = np.array(image)
            mask_array = np.array(mask)
            overlay[mask_array > 0] = [255, 0, 0]  # Red overlay
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        else:
            # For non-segmentation tasks, just show the image
            plt.figure(figsize=(6, 6))
            plt.imshow(sample['image'])
            plt.title(f"Sample {i+1}")
            plt.axis('off')
            plt.show()
        
        print("-" * 50)

