import google.generativeai as genai
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import time  # Import the time module

API_KEY = "YOUR_API_KEY_HERE"
genai.configure(api_key=API_KEY)

predefined_prompt = "Please describe what you see in this gastrointestinal imaging scan using simple, everyday language. Focus only on the visual elements without making any medical interpretations, diagnoses, or clinical assessments. Describe the image as one continuous passage, including: The overall appearance and coloring of the tissues and structures Any visible structures, shapes, or patterns you can see The location of different features (left side, right side, upper area, lower area, center, etc.) Any areas that appear lighter, darker, or different in texture The general size and position of visible elements Any text, labels, or annotations visible on the image Use descriptive words that anyone could understand, avoid technical medical terms, and simply report what is visually apparent in the image. Present your description as a single paragraph without bullet points or separate sections."

model = genai.GenerativeModel('gemma-3-27b-it')

image_directory_path = "data/images"
image_path = Path(image_directory_path)


# List to store the results
results = []
image_files = list(image_path.glob('*'))

if not image_files:
    print(f"No images found in the directory: {image_directory_path}")
else:
    print(f"Found {len(image_files)} images to process.")

    for img_file in image_files:
        print(f"Processing: {img_file.name}")
        try:
            # Open the image file
            img = Image.open(img_file)

            # Generate content with the image and prompt
            response = model.generate_content([predefined_prompt, img])

            print(response.text)

            # Store the image ID and the generated description
            results.append({
                'img_id': img_file.name,
                'visual_description': response.text
            })

            # --- RATE LIMITING ---
            # Wait for 6 seconds before the next request to stay within the 10 requests per minute limit.
            print("Waiting 6 seconds before next request...")
            time.sleep(5)

        except Exception as e:
            print(f"An error occurred while processing {img_file.name}: {e}")

# Create a pandas DataFrame from the results
if results:
    df = pd.DataFrame(results)

    # Display the first few rows of the new dataset
    print("\n--- Generated Dataset ---")
    print(df.head())

    # Save the DataFrame to a CSV file
    output_csv_path = "visual_descriptions.csv"
    df.to_csv(output_csv_path, index=False)
    print(f"\nDataset successfully saved to: {output_csv_path}")
else:
    print("\nNo descriptions were generated.")