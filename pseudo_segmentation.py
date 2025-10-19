import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
import torch
import csv
import os
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

#PSEUDO SEGMENTATION SCRIPT##

IMG_DIR = "data/images"

# Load processor and model
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# Choose device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to device
model.to(device)

def paste_mask_on_original(mask, coords, original_shape, feather=20):
    """
    Resize predicted mask back and paste it on full-size image
    with smooth edge interpolation (feathering).
    
    feather = number of pixels to fade at edges
    """
    y1, y2, x1, x2 = coords
    full_mask = np.zeros(original_shape[:2], dtype=np.float32)

    # Resize predicted mask to ROI size
    resized = cv2.resize(mask, (x2 - x1, y2 - y1))

    # --- build feather mask ---
    h, w = resized.shape
    yy, xx = np.ogrid[:h, :w]

    # distance to nearest edge
    dist_y = np.minimum(yy, h - 1 - yy)
    dist_x = np.minimum(xx, w - 1 - xx)
    dist = np.minimum(dist_y, dist_x)

    # normalize to [0,1] with feather region
    weight = np.clip(dist / feather, 0, 1).astype(np.float32)

    # apply weight
    smoothed = resized * weight

    # paste into full mask
    full_mask[y1:y2, x1:x2] = smoothed
    return full_mask

def create_absolute_mask(segmented_img, thresh_val=128, kernel_size=5, auto_otsu=False, relative_strict=True, percentile=80):
    """
    Convert segmented/probability mask into a clean absolute mask.
    Strict mode keeps only the brightest pixels.
    """
    import numpy as np
    import cv2

    # Ensure grayscale
    if len(segmented_img.shape) == 3:
        gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = segmented_img.copy()

    # If float (0–1), scale to 0–255
    if gray.dtype in [np.float32, np.float64]:
        gray = (gray * 255).astype(np.uint8)

    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    # Strict relative threshold
    if relative_strict:
        t_val = np.percentile(gray, percentile)  # e.g., 95th percentile
        _, binary = cv2.threshold(gray, t_val, 255, cv2.THRESH_BINARY)
    elif auto_otsu:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# Load CSV
abnormility="z-line"
df = pd.read_csv(f"data/aux_data/landmark_{abnormility}-updated.csv")

#df = df.sample(frac=1).reset_index(drop=True)
# Preview first 10 rows
tests = df

# Create output directory if it doesn't exist
MASK_DIR = f"your/path/masks"
os.makedirs(MASK_DIR, exist_ok=True)

# Prepare list to collect metadata rows
mask_metadata = []

# --- main loop ---
for img_id, loc, colors in zip(tests['img_id'], tests['landmark_location'], tests['landmark_color']):
    if pd.isna(loc):
        print(f"Skipping {img_id} due to missing location")
        continue

    prompts = ["hole"]

    # Load full image
    full_img = cv2.imread(f"{IMG_DIR}/{img_id}.jpg")
    full_rgb = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)

    # Whole image inference
    inputs = processor(
        text=prompts,
        images=[full_rgb] * len(prompts),
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    preds = outputs.logits.unsqueeze(1)

    full_masks = []
    for i, prompt in enumerate(prompts):
        mask = torch.sigmoid(preds[i][0]).cpu().numpy()
        abs_mask = create_absolute_mask(mask, relative_strict=True, kernel_size=3)
        resized_mask = cv2.resize(abs_mask, (full_rgb.shape[1], full_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        full_masks.append(resized_mask)

        # Save mask to file
        mask_filename = f"{img_id}_{abnormility}_{i}.jpg"
        mask_path = os.path.join(MASK_DIR, mask_filename)
        cv2.imwrite(mask_path, resized_mask)  # Convert to 0–255 grayscale

        # Record metadata
        mask_id = f"{img_id}_{abnormility}_mask_{i}"
        mask_metadata.append({
            "img_id": img_id,
            "landmark_location": loc,
            "landmark_color": colors,
            "mask_id": mask_id
        })

# --- Save metadata to CSV ---
metadata_csv_path = f"{abnormility}_masks_metadata.csv"
with open(metadata_csv_path, mode="w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["img_id", "landmark_location", "landmark_color", "mask_id"])
    writer.writeheader()
    writer.writerows(mask_metadata)

# Load CSV
abnormility="cecum"
df = pd.read_csv(f"data/aux_data/landmark_{abnormility}-updated.csv")

df = df.sample(frac=1).reset_index(drop=True)
# Preview first 10 rows
tests = df

# Prepare list to collect metadata rows
mask_metadata = []

# --- main loop ---
for img_id, loc, colors in zip(tests['img_id'], tests['landmark_location'], tests['landmark_color']):
    if pd.isna(loc):
        print(f"Skipping {img_id} due to missing location")
        continue

    prompts = ["hole", "tunnel hole"]

    # Load full image
    full_img = cv2.imread(f"{IMG_DIR}/{img_id}.jpg")
    full_rgb = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)

    # Whole image inference
    inputs = processor(
        text=prompts,
        images=[full_rgb] * len(prompts),
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    preds = outputs.logits.unsqueeze(1)

    full_masks = []
    for i, prompt in enumerate(prompts):
        mask = torch.sigmoid(preds[i][0]).cpu().numpy()
        abs_mask = create_absolute_mask(mask, relative_strict=True, kernel_size=3)
        resized_mask = cv2.resize(abs_mask, (full_rgb.shape[1], full_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        full_masks.append(resized_mask)

    merged_mask = np.zeros_like(full_masks[0])
    for m in full_masks:
        merged_mask = np.logical_or(merged_mask, m)

    merged_mask = (merged_mask * 255).astype(np.uint8)
    mask_filename = f"{img_id}_{abnormility}_mask_{0}.jpg"
    mask_path = os.path.join(MASK_DIR, mask_filename)
    cv2.imwrite(mask_path, merged_mask)
    mask_id = f"{img_id}_{abnormility}_mask_0"
    mask_metadata.append({
            "img_id": img_id,
            "abnormality_location": loc,
            "abnormality_color": colors,
            "mask_id": mask_id
    })

# --- Save metadata to CSV ---
metadata_csv_path = f"{abnormility}_masks_metadata.csv"
with open(metadata_csv_path, mode="w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["img_id", "landmark_location", "landmark_color", "mask_id"])
    writer.writeheader()
    writer.writerows(mask_metadata)


abnormility="ulcerative_colitis"
df = pd.read_csv(f"data/aux_data/landmark_{abnormility}-updated.csv")

df = df.sample(frac=1).reset_index(drop=True)
# Preview first 10 rows
tests = df

# Prepare list to collect metadata rows
mask_metadata = []

# --- main loop ---
for img_id, loc, colors in zip(tests['img_id'], tests['abnormility_location'], tests['abnormility_color']):
    if pd.isna(loc) or not isinstance(colors, str):
        print(f"Skipping {img_id} due to missing location")
        continue

    prompts = []
    color_list = colors.split(';')
    color_list = [col.strip() for col in color_list]
    
    prompt1 = []
    for color in color_list:
        prompt1.append(f"{color} spots")
    
    prompt2 = []
    for color in color_list:
        prompt2.append(f"{color} scars")
    

    if "red" in color_list:
        prompt1.append("red blood, blood stains")
        prompt2.append("red blood, blood stains")


    #     # If you want to convert them to strings:
    prompt1_str = ', '.join(prompt1)
    prompt2_str = ', '.join(prompt2)

    prompts.append(prompt1_str)
    prompts.append(prompt2_str)
    prompts.append("rough surface")
    prompts.append("hole")

    # Load full image
    full_img = cv2.imread(f"{IMG_DIR}/{img_id}.jpg")
    full_rgb = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)

    # Whole image inference
    inputs = processor(
        text=prompts,
        images=[full_rgb] * len(prompts),
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    preds = outputs.logits.unsqueeze(1)

    full_masks = []
    for i, prompt in enumerate(prompts):
        mask = torch.sigmoid(preds[i][0]).cpu().numpy()
        abs_mask = create_absolute_mask(mask, relative_strict=True, kernel_size=3)
        resized_mask = cv2.resize(abs_mask, (full_rgb.shape[1], full_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        full_masks.append(resized_mask)

    merged_mask = np.zeros_like(full_masks[0])
    for m in full_masks:
        merged_mask = np.logical_or(merged_mask, m)

    merged_mask = (merged_mask * 255).astype(np.uint8)
    mask_filename = f"{img_id}_{abnormility}_mask_{0}.jpg"
    mask_path = os.path.join(MASK_DIR, mask_filename)
    cv2.imwrite(mask_path, merged_mask)
    mask_id = f"{img_id}_{abnormility}_mask_0"
    mask_metadata.append({
            "img_id": img_id,
            "abnormality_location": loc,
            "abnormality_color": colors,
            "mask_id": mask_id
    })

# --- Save metadata to CSV ---
metadata_csv_path = f"{abnormility}_masks_metadata.csv"
with open(metadata_csv_path, mode="w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["img_id", "abnormality_location", "abnormality_color", "mask_id"])
    writer.writeheader()
    writer.writerows(mask_metadata)

abnormility="oesophagitis"
df = pd.read_csv(f"data/aux_data/landmark_{abnormility}-updated.csv")

df = df.sample(frac=1).reset_index(drop=True)
# Preview first 10 rows
tests = df

# Prepare list to collect metadata rows
mask_metadata = []

# --- main loop ---
for img_id, loc, colors in zip(tests['img_id'], tests['abnormility_location'], tests['abnormility_color']):
    if pd.isna(loc) or not isinstance(colors, str):
        print(f"Skipping {img_id} due to missing location")
        continue

    prompts = []
    color_list = colors.split(';')
    color_list = [col.strip() for col in color_list]
    
    prompt1 = []
    for color in color_list:
        prompt1.append(f"{color} spots")
    
    prompt2 = []
    for color in color_list:
        prompt2.append(f"{color} scars")
    

    if "red" in color_list:
        prompt1.append("red blood, blood stains")
        prompt2.append("red blood, blood stains")


    #     # If you want to convert them to strings:
    prompt1_str = ', '.join(prompt1)
    prompt2_str = ', '.join(prompt2)

    prompts.append(prompt1_str)
    prompts.append(prompt2_str)
    prompts.append("rough surface")
    prompts.append("hole")

    # Load full image
    full_img = cv2.imread(f"{IMG_DIR}/{img_id}.jpg")
    full_rgb = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)

    # Whole image inference
    inputs = processor(
        text=prompts,
        images=[full_rgb] * len(prompts),
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    preds = outputs.logits.unsqueeze(1)

    full_masks = []
    for i, prompt in enumerate(prompts):
        mask = torch.sigmoid(preds[i][0]).cpu().numpy()
        abs_mask = create_absolute_mask(mask, relative_strict=True, kernel_size=3)
        resized_mask = cv2.resize(abs_mask, (full_rgb.shape[1], full_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        full_masks.append(resized_mask)

    merged_mask = np.zeros_like(full_masks[0])
    for m in full_masks:
        merged_mask = np.logical_or(merged_mask, m)

    merged_mask = (merged_mask * 255).astype(np.uint8)
    mask_filename = f"{img_id}_{abnormility}_mask_{0}.jpg"
    mask_path = os.path.join(MASK_DIR, mask_filename)
    cv2.imwrite(mask_path, merged_mask)
    mask_id = f"{img_id}_{abnormility}_mask_0"
    mask_metadata.append({
            "img_id": img_id,
            "abnormality_location": loc,
            "abnormality_color": colors,
            "mask_id": mask_id
    })

# --- Save metadata to CSV ---
metadata_csv_path = f"{abnormility}_masks_metadata.csv"
with open(metadata_csv_path, mode="w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["img_id", "abnormality_location", "abnormality_color", "mask_id"])
    writer.writeheader()
    writer.writerows(mask_metadata)



##MASK CLEANING AND SIMPLIFICATION SCRIPT##

def process_mask(orignal_image, mask_image, min_area=70000, simplify_epsilon=0.009, tag=None):
    image_array = np.array(orignal_image)
    mask_array = np.array(mask_image)
    image_height, image_width = image_array.shape[:2]

    # --- Robustly get a single-channel mask ---
    if mask_array.ndim == 3:
        mask_gray = np.max(mask_array[..., :3], axis=2)
    else:
        mask_gray = mask_array.copy()

    binary_mask = (mask_gray > 0).astype(np.uint8) * 255

    # --- Check the original image left-quarter for dark pixels ---
    if image_array.ndim == 3:
        img_gray = cv2.cvtColor(image_array[..., :3], cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image_array.copy()

    box_x1, box_x2 = 0, image_width // 4
    artifact_box = img_gray[:, box_x1:box_x2]

    dark_thresh = 20  # tunable
    dark_pixels_mask = (artifact_box < dark_thresh)
    binary_mask[:, box_x1:box_x2][dark_pixels_mask] = 0

    # --- Remove areas in original image that match color #00bd91 ---
    target_color = np.array([0, 189, 145])  # RGB for #00bd91
    tolerance = 40  # +/- tolerance for color similarity (tune as needed)

    if image_array.ndim == 3:
        diff = np.linalg.norm(image_array[..., :3] - target_color, axis=2)
        color_mask = diff < tolerance
        binary_mask[color_mask] = 0  # remove overlapping mask pixels

    # --- Find contours ---
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    vis_data = []
    for contour in contours:
        area = cv2.contourArea(contour)

        if tag and ("cecum" in tag or "z-line" in tag):
            min_area = 35000
        
        if area < min_area:
            continue

        # Simplify contour
        perimeter = cv2.arcLength(contour, True)
        epsilon = simplify_epsilon * perimeter  # adaptive simplification
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)

        # Convert to polygon format
        polygon = simplified_contour.reshape(-1, 2)
        # Skip if polygon has less than 3 points
        if len(polygon) < 3:
            continue

        # bounding box not needed for now
        vis_data.append(polygon)

    return vis_data


# --- Paths ---
IMG_DIR = r"data\images"
OUTPUT_DIR = r"your/path/pseudo_masks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Main loop ---
for mask_file in os.listdir(MASK_DIR):
    if not mask_file.lower().endswith((".jpg", ".png")):
        continue

    # Get image id from mask file (e.g., img1_object_mask_0.jpg -> img1)
    img_id = mask_file.split("_")[0]
    tag = mask_file.split("_")[1]

    mask_path = os.path.join(MASK_DIR, mask_file)
    img_path = os.path.join(IMG_DIR, f"{img_id}.jpg")

    if not os.path.exists(img_path):
        print(f"⚠️ Skipping {mask_file}, no matching original image found.")
        continue

    # Load original and mask
    original_img = Image.open(img_path).convert("RGB")
    mask_img = Image.open(mask_path).convert("RGB")

    # Process
    polygons = process_mask(original_img, mask_img, tag=tag)

    # Make a new blank mask
    new_mask = np.zeros((original_img.height, original_img.width), dtype=np.uint8)

    # Draw simplified polygons
    for poly in polygons:
        cv2.fillPoly(new_mask, [poly.astype(np.int32)], 255)

    # Save new simplified mask
    out_path = os.path.join(OUTPUT_DIR, mask_file)
    cv2.imwrite(out_path, new_mask)

    print(f"✅ Saved simplified mask: {out_path}")