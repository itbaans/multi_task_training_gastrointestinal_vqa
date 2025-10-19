import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

def get_prompt(Q_A_PAIRS, VIS_DISC, NATURAL_QUESTION, NATURAL_ANSWER):
    return f"""You are tasked with creating enhanced natural answers for medical images. Your job is to take a brief natural answer and expand it with relevant details from the Q/A pairs and visual description, while staying strictly focused on the topics mentioned in the original natural answer.

## Instructions:
1. **Stay focused**: Only enhance details that directly relate to what's mentioned in the original natural answer
2. **Use Q/A pairs**: Extract specific clinical facts (sizes, types, locations, colors, counts)
3. **Incorporate visual description**: Add descriptive visual details that support the topics in the original answer
4. **Do not add unrelated information**: If the original answer is about text, don't mention polyps. If it's about abnormalities, don't mention unrelated text details.

## Examples:

### Example 1:
**Natural Question:** Is there any textual content present?
**Natural Answer:** textual content observed

**Ground Q/A Pairs:**
- What type of polyp is present? → none
- Where in the image is the abnormality? → center; center-left; center-right; lower-center; lower-left; lower-right; upper-center; upper-left; upper-right
- Are there any abnormalities in the image? Check all that are present. → ulcerative colitis
- What is the size of the polyp? → none
- How many polyps are in the image? → 0
- Where in the image is the instrument? → none
- What type of procedure is the image taken from? → colonoscopy
- Have all polyps been removed? → not relevant
- Is there text? → yes
- Where in the image is the anatomical landmark? → none
- How many instrumnets are in the image? → 0
- Are there any anatomical landmarks in the image? Check all that are present. → none
- What color is the abnormality? If more than one separate with ; → pink; red; white
- Is there a green/black box artefact? → no
- Are there any instruments in the image? Check all that are present. → none

**Visual Description:** The image is mostly filled with a reddish-pink, somewhat bumpy surface that appears to have a lot of folds and ridges running across it. The coloring is generally consistent, but there are areas that appear lighter, almost yellowish-white, particularly in the upper left corner and scattered throughout as small, shiny spots. The texture seems uneven, with some areas looking smoother and others more rough or raised. Towards the center-right, there’s a larger, brighter white area that looks a bit more rounded and puffy compared to the surrounding reddish tissue. The overall shape of the visible area is roughly octagonal, with the reddish-pink surface filling most of it. Along the left edge, there are some lighter orange-yellow tones. In the upper left corner, there’s some white text and numbers: “21/06/2012”, “12:24:03”, “CVP:7”, “G:N”, and “Im:A5”.
**Enhanced Natural Answer:** "Yes, textual content is observed on the image. White text and numbers are present in the upper left corner, including the details '21/06/2012', '12:24:03', 'CVP:7', 'G:N', and 'Im:A5'."

### Example 2:
**Natural Question:** Where is the abnormality located in the image?
**Natural Answer:** abnormality located in central region

**Ground Q/A Pairs:**
- Where in the image is the abnormality? → center
- Have all polyps been removed? → no
- What type of procedure is the image taken from? → colonoscopy
- Are there any instruments in the image? Check all that are present. → none
- Are there any anatomical landmarks in the image? Check all that are present. → none
- Where in the image is the instrument? → none
- What type of polyp is present? → paris iia
- What is the size of the polyp? → <5mm
- How many polyps are in the image? → 1
- Are there any abnormalities in the image? Check all that are present. → polyp
- Is there a green/black box artefact? → yes
- What color is the abnormality? If more than one separate with ; → pink; red
- Where in the image is the anatomical landmark? → none
- Is there text? → no
- How many instrumnets are in the image? → 0

**Visual Description:** The image is mostly filled with a reddish-pink, somewhat uneven surface that appears soft and folded, like a crumpled cloth. The coloring is generally consistent across the majority of the view, though it gets brighter towards the upper right corner, almost appearing white and glowing. In the center of the image, there’s a slightly lighter, circular area that looks smoother than the surrounding texture. Scattered throughout the image, and more concentrated near the center, are numerous small, thin, white shapes that resemble tiny slivers or fragments. Along the lower edge, there’s a darker, black area that takes up about a quarter of the image’s width. In the lower right corner, there’s a bright white, rounded shape. On the left side, near the top, there’s a small, dark marking with the number “15” next to it. Additionally, there are a few small, square markings and a curved symbol in the lower left corner. The overall shape of the image is an octagon, with the view appearing to be taken from inside a tube or cavity.

**Enhanced Natural Answer:** "The abnormality, a polyp, is located in the center of the image. It appears as a slightly lighter, circular area that looks smoother than the surrounding reddish-pink, folded texture."

### Example 3:
**Natural Question:** Were all identified polyps successfully excised?
**Natural Answer:**  some polyps remain present

**Ground Q/A Pairs:**
- Are there any abnormalities in the image? Check all that are present. → polyp
- Have all polyps been removed? → no
- Is there a green/black box artefact? → yes
- Is there text? → yes
- What type of procedure is the image taken from? → colonoscopy
- What is the size of the polyp? → >20mm
- What type of polyp is present? → paris ip
- What color is the abnormality? If more than one separate with ; → pink; red
- How many polyps are in the image? → 1
- Where in the image is the abnormality? → center; center-right; lower-center; lower-right; upper-center; upper-right

**Visual Description:** The image appears to be a view from inside a tube-like structure, with a generally pinkish-red coloring throughout. The majority of the view is taken up by a large, rounded shape positioned towards the right side and slightly above the center. This shape is a brighter, more intense red than the surrounding areas and has a bumpy, textured surface. To the left of this large shape, the pinkish-red color continues, appearing somewhat smoother in texture, and fading into a darker area on the far left edge of the image. Along the upper edge, there’s a darker band that curves across, and a similar darker band is visible along the lower edge. In the upper left corner, there are several lines of white text and numbers, and a black rectangular shape obscures a portion of the lower left corner. The overall impression is of a circular or oval opening with a prominent, textured feature inside.
**Enhanced Natural Answer:** ""No, not all polyps have been excised. One large Paris Ip type polyp, measuring greater than 20mm, remains present. It is located predominantly on the right side and central areas of the image, appearing as a large, rounded, and bumpy mass with a brighter, more intense red coloration compared to the surrounding tissue."

### Your Test Case
*Natural Language Question*: {NATURAL_QUESTION}

*Natural Language Answer*: {NATURAL_ANSWER}

*Q/A Pairs*:
{Q_A_PAIRS}

*Visual Description*:
{VIS_DISC}

## Your Response (avoid using verbose specific locations and also avoid mentioning any meta data like q/a pairs or visual description directly in response):
"""

import os
import re
import csv
import time  # Import the time module for delays
import google.generativeai as genai

from tqdm import tqdm


REQUESTS_PER_MINUTE = 4
# Set how often to save the CSV to disk (e.g., every 10 items)
SAVE_INTERVAL = 10

# Calculate the required delay between requests in seconds
DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE

# Configure the Generative AI client with your API key
try:
    genai.configure(api_key="API_KEY_HERE")
except Exception as e:
    print(f"Error configuring Google AI: {e}")
    exit()

try:
    model = genai.GenerativeModel('gemma-3-27b-it')
    print(f"Successfully initialized model: gemma-3-27b-it")
    print(f"Rate limit set to {REQUESTS_PER_MINUTE} req/min. Delay between requests: {DELAY_BETWEEN_REQUESTS:.2f} seconds.")
    print(f"CSV will be saved to disk every {SAVE_INTERVAL} items.")
except Exception as e:
    print(f"Error initializing the model: {e}")
    exit()


def get_model_response(prompt):

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        # Use tqdm.write to print without disrupting the progress bar
        tqdm.write(f"\n--- An error occurred during the API call: {e} ---")
        return "API_ERROR"

def count_prompt_tokens(prompt_text):

    try:
        token_count = model.count_tokens(prompt_text)
        return token_count.total_tokens
    except Exception as e:
        print(f"An error occurred while counting tokens: {e}")
        return -1


# --- Step 3: Main Parsing and Processing Logic (Updated) ---

def parse_and_process_file(input_file_path, output_csv_path):
    """
    Reads a text file, parses each item, gets a model response with rate limiting,
    and writes the results to a CSV file with periodic saves.
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{input_file_path}' was not found.")
        return

    item_chunks = content.split('Image ID:')[1:]

    if not item_chunks:
        print("No items found in the file.")
        return

    print(f"Found {len(item_chunks)} items. Starting processing...")

    # Open the CSV file to write the results
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['img_id', 'n_question', 'n_answer', 'response'])

        # Use enumerate with start=1 for easier interval checking
        for i, chunk in enumerate(tqdm(item_chunks, desc="Processing items"), start=1):
            full_chunk = 'Image ID:' + chunk
            img_id_line = chunk.strip().split('\n')[0]
            img_id = img_id_line.strip()
            
            pattern = re.compile(
                r"Natural Question:\s*(.*?)\n"
                r"Natural Answer:\s*(.*?)\n"
                r"Ground Q/A Pairs:\s*(.*?)"
                r"Visual Description:",
                re.DOTALL
            )
            match = pattern.search(full_chunk)

            if match:
                natural_question = match.group(1).strip()
                natural_answer = match.group(2).strip()
                q_a_pairs = match.group(3).strip()
                
                vis_desc_start = full_chunk.find("Visual Description:")
                vis_disc = full_chunk[vis_desc_start + len("Visual Description:"):].strip()

                prompt = get_prompt(Q_A_PAIRS=q_a_pairs, VIS_DISC=vis_disc, NATURAL_QUESTION=natural_question, NATURAL_ANSWER=natural_answer)
                model_response = get_model_response(prompt)
                print(model_response)
                print(count_prompt_tokens(prompt))
                csv_writer.writerow([img_id, natural_question, natural_answer, model_response])

                if i % SAVE_INTERVAL == 0:
                    csvfile.flush()  # Force write from buffer to disk
                    # Use tqdm.write to avoid messing up the progress bar
                    tqdm.write(f"--- Checkpoint: Saved progress to '{output_csv_path}' at item {i} ---")

            else:
                tqdm.write(f"\nWarning: Could not parse item #{i}. Skipping.")

                time.sleep(DELAY_BETWEEN_REQUESTS)
    
    print(f"\nProcessing complete. All results saved to '{output_csv_path}'")

input_filename = 'data/combined_samples.txt'
output_filename = 'your/path/texual_expl.csv'
    
parse_and_process_file(input_filename, output_filename)