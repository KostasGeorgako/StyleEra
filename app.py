from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Query
import os
import base64
import uvicorn
import json
from collections import defaultdict
import ast
import re
import asyncio
import random
import numpy as np
import requests
from pydantic import BaseModel

# for clip
import torch
import clip
from PIL import Image
import torch.nn.functional as F

# for gtp-04
from openai import AzureOpenAI

from fastapi.middleware.cors import CORSMiddleware

from psearch import PinterestScraper
from typing import List, Dict

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend URL like ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# initialize the scraper
scraper = PinterestScraper()

BATCH_SIZE = 3

# Azure OpenAI Client Setup
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Prompts
ANALYZE_IMAGE_PROMPT = '''
You are a fashion and style expert AI.

You will be given an image of an outfit. Your task is to analyze the outfit and generate a detailed yet structured fashion description in JSON format.

INSTRUCTIONS:
1. Identify fashion/style tags that describe the image. These may include:
   - Styles or aesthetics (e.g., "grunge", "minimalist", "streetwear", "bohemian")
   - Materials or textures (e.g., "denim", "leather", "knit")
   - Colors (e.g., "earth tones", "monochrome")
   - Patterns (e.g., "patchwork", "striped")

2. For each tag, assign a confidence score between **0.0 and 1.0**, representing how confidently the tag applies to the image.

3. Estimate the most likely **era range** (e.g. "1970-1980").

OUTPUT FORMAT:
Return **only** a JSON object, with the following structure:

{
  "tags": {
    "tag1": confidence1,
    "tag2": confidence2,
    ...
  },
  "estimatedEra": "YYYY-YYYY"
}

STRICT RULES:
- Do not explain your answer.
- Do not include any text outside the JSON.
- Do not wrap the output in markdown (no ```json or ```).
- Ensure valid JSON (use double quotes for keys and strings).
'''

BATCH_ANALYZE_IMAGE_PROMPT = '''
You are a fashion and style expert AI.

You will be shown multiple images of outfits. Each image is labeled in order (e.g., "Image 1", "Image 2", etc.). Your task is to analyze **each image individually** and produce a structured JSON result for each.

For each image, do the following:
1. Identify fashion/style tags that describe the outfit. These may include:
   - Styles or aesthetics (e.g., "grunge", "minimalist", "streetwear", "bohemian")
   - Materials or textures (e.g., "denim", "leather", "knit")
   - Colors (e.g., "earth tones", "monochrome")
   - Patterns (e.g., "patchwork", "striped")

2. For each tag, assign a confidence score between **0.0 and 1.0**.

3. Estimate the most likely **era range** (e.g., "1970-1980").

Return a **single JSON object** mapping each image label to its result. The structure should look like this:

{
  "Image 1": {
    "tags": {
      "tag1": confidence1,
      "tag2": confidence2,
      ...
    },
    "estimatedEra": "YYYY-YYYY"
  },
  "Image 2": {
    "tags": {
      ...
    },
    "estimatedEra": ...
  },
  ...
}

STRICT RULES:
- Do not explain anything.
- Return only a raw JSON object.
- Do not wrap the output in markdown (no ```).
- Use valid JSON formatting (double quotes, properly nested).
'''


FUSE_TAGS_PROMPT = '''
You are a fashion domain expert specializing in taxonomy and tagging systems.

You are given a list of fashion or style-related tags. Your task is to **group only extremely similar tags** (i.e., referring to the same specific fashion item, material, or concept). For each group, select a **canonical tag** — a common, lowercase tag from the group — and map the original tags to it.

REQUIREMENTS:
- Do **not** group tags that are only vaguely related.
- Do **not** invent new canonical tags — reuse one of the tags from the group.
- Canonical tags must be lowercase and should match common fashion terminology.
- All input tags **must appear in the output**, either individually or within a group.
- Ensure each tag is mapped once and only once.

OUTPUT:
Return a single JSON object mapping canonical tags to lists of their corresponding original tags. For example:

{
  "boho chic": ["boho chic", "bohemian"],
  "hippie": ["hippie"],
  "cottagecore": ["cottagecore", "vintage inspired"],
  "maxi skirt": ["maxi skirt", "long flowing skirt"]
}

STRICT RULES:
- Output only a raw JSON object.
- Do not include explanations.
- Do not wrap the JSON in markdown (e.g., no ```json).
- Use valid JSON formatting (double quotes, proper nesting).

Now group the following tags:
Input tags:
'''



def safe_parse_json(possible_json: str) -> dict:
    print(possible_json)
    # Try to extract JSON from markdown-style ```json blocks
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", possible_json, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = possible_json.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")

def build_decade_distribution(era_ranges: list[str]) -> dict:
    decade_scores = defaultdict(float)

    for era in era_ranges:
        try:
            start, end = map(int, era.split('-'))
            decades = list(range(start // 10 * 10, end + 1, 10))
            weight = 1 / len(decades)
            for decade in decades:
                decade_label = f"{decade}s"
                decade_scores[decade_label] += weight
        except Exception:
            continue  # skip malformed input

    total = sum(decade_scores.values())
    if total == 0:
        return {}

    return {k: round(v / total, 3) for k, v in decade_scores.items()}


async def azure_gpt4o_get_image_style_batch(batch: list[tuple[int, UploadFile]]) -> list[tuple[int, dict, bytes]]:
    try:
        image_entries = []
        file_contents = {}
        index_to_id = {}

        # Add each image to content list
        for idx, (file_id, file) in enumerate(batch):
            file_content = await file.read()
            file_contents[idx] = file_content
            index_to_id[idx] = file_id

            image_data = base64.b64encode(file_content).decode("utf-8")
            mime_type = file.content_type

            image_entries.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
            })

        # Add the instruction text at the end
        image_entries.append({
            "type": "text",
            "text": BATCH_ANALYZE_IMAGE_PROMPT
        })

        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                {
                    "role": "user",
                    "content": image_entries
                }
            ],
            max_tokens=2048,
        )

        content = response.choices[0].message.content
        result_json = safe_parse_json(content)

        # Turn "Image 1", "Image 2", ... back into (file_id, data, file_content)
        output = []
        for i in range(len(batch)):
            label = f"Image {i+1}"
            if label not in result_json:
                raise ValueError(f"Missing analysis for {label}")
            file_id = index_to_id[i]
            file_content = file_contents[i]
            output.append((file_id, result_json[label], file_content))

        return output

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Azure OpenAI batch error: {str(e)}")



# Azure GPT-4o helper to analyze image
async def azure_gpt4o_get_image_style(file_id: int, file: UploadFile) -> tuple:
    try:
        file_content = await file.read()
        image_data = base64.b64encode(file_content).decode("utf-8")
        mime_type = file.content_type

        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
                        {"type": "text", "text": ANALYZE_IMAGE_PROMPT}
                    ]
                }
            ],
            max_tokens=1024,
        )

        content = response.choices[0].message.content
        data = safe_parse_json(content)
        return (file_id, data, file_content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Azure OpenAI error: {str(e)}")

# Azure GPT-4o helper to fuse tags
async def azure_gpt4o_get_fused_tags(tags: list[str]) -> dict:
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                {
                    "role": "user",
                    "content": f"{FUSE_TAGS_PROMPT.strip()}\n\nInput tags:\n{json.dumps(tags, indent=2)}"
                }
            ],
            max_tokens=1024,
        )
        content = response.choices[0].message.content
        data = safe_parse_json(content)

        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Azure OpenAI fusion error: {str(e)}")

# CLIP image encoder
async def clip_encode_image(file_bytes: bytes):
    image = Image.open(BytesIO(file_bytes)).convert("RGB")
    image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
    return image_features / image_features.norm(dim=-1, keepdim=True)

# CLIP text encoder
def clip_encode_text(tags):
    text_tokens = clip.tokenize(tags).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
    return text_features / text_features.norm(dim=-1, keepdim=True)

# Cosine similarity
def cosine_similarity(image_features, text_features):
    return (image_features @ text_features.T).squeeze(0)

@app.post("/get_analysis")
async def merge_image_styles(files: list[UploadFile] = File(...)) -> dict:
    file_indices = list(enumerate(files))
    batches = [file_indices[i:i+BATCH_SIZE] for i in range(0, len(file_indices), BATCH_SIZE)]

    tasks = [azure_gpt4o_get_image_style_batch(batch) for batch in batches]
    batch_results = await asyncio.gather(*tasks)
    flat_results = [item for batch in batch_results for item in batch]

    merged = {"uploadedTotal": len(flat_results), "tags": {}, "eras": []}
    all_tags = set()
    original_infos = {}

    for file_id, data, file_bytes in flat_results:
        tags = data['tags']
        era = data.get("estimatedEra")
        if era:
            merged["eras"].append(era)

        original_infos[file_id] = {
            "bytes": file_bytes,
            "tags": list(tags.keys()),
            "scores": list(tags.values())
        }

        all_tags.update(tags.keys())

    # fuse tags
    fused_tags = await azure_gpt4o_get_fused_tags(list(all_tags))

    # build mappings from the fused tags response
    orig_to_canonical = {}
    canonical_freq = {}
    for canonical, originals in fused_tags.items():
        for original in originals:
            orig_to_canonical[original] = canonical
        canonical_freq[canonical] = canonical_freq.get(canonical, 0) + 1

    # CLIP encode in parallel
    image_encodings = await asyncio.gather(*(clip_encode_image(info["bytes"]) for info in original_infos.values()))

    # unique tag embeddings
    unique_tags = list(set(tag for info in original_infos.values() for tag in info["tags"]))
    tag_to_embedding = {}
    text_features = clip_encode_text(unique_tags)

    for i, tag in enumerate(unique_tags):
        tag_to_embedding[tag] = text_features[i]

    # compute similarities and distributions
    tags_distribution = {}

    for img_embedding, (file_id, info) in zip(image_encodings, original_infos.items()):
        for tag, score in zip(info["tags"], info["scores"]):
            if tag not in orig_to_canonical:
                continue

            canonical = orig_to_canonical[tag]
            text_emb = tag_to_embedding[tag].unsqueeze(0)
            sim = cosine_similarity(img_embedding, text_emb).item()

            # scoring = confidence * similarity * frequency
            contrib = 0.7 * ( score * canonical_freq[canonical] ) + 0.3 * sim

            # if it already exists add it
            tags_distribution[canonical] = tags_distribution.get(canonical, 0) + contrib


    merged["tags"] = tags_distribution
    # dont forget to compute the decade era distribution
    merged["eras"] = build_decade_distribution(merged["eras"])

    # normalize the distributions
    tags_total = sum(merged["tags"].values())
    if tags_total > 0:
        merged["tags"] = {k: v / tags_total for k, v in merged["tags"].items()}

    eras_total = sum(merged["eras"].values())
    if eras_total > 0:
        merged["eras"] = {k: v / eras_total for k, v in merged["eras"].items()}


    print(json.dumps(merged, indent=4))

    return merged


import random

@app.post("/generate_searches")
def generate_searches(
    total_samples: int = Query(1, ge=1, le=20),
    distributions: dict = Body(...)
):

    tags_dist = distributions.get('tags', {})
    era_dist = distributions.get('eras', {})

    if not tags_dist or not era_dist:
        return {"error": "Both 'tags' and 'eras' distributions must be provided"}

    tag_keys = list(tags_dist.keys())
    tag_weights = np.array(list(tags_dist.values()))

    if total_samples > len(tag_keys):
        return {"error": f"Cannot sample {total_samples} unique tags from only {len(tag_keys)} available"}

    sampled_tags = np.random.choice(tag_keys, size=total_samples, replace=False, p=tag_weights)
    sampled_tags = sorted(sampled_tags, key=lambda tag: -tags_dist[tag])

    # For era, just pick the most likely one, or use weighted random choice with only 1 pick
    era_keys = list(era_dist.keys())
    era_weights = np.array(list(era_dist.values()))
    sampled_era = np.random.choice(era_keys, p=era_weights)

    return {
        "tags": [str(tag) for tag in sampled_tags],
        "era": str(sampled_era)
    }

@app.post("/get_recommendations")
def get_recommendations(
    search_term: str = Query(default="fashion"),
    scrolls: int = Query(1, ge=1, le=5),
    top_n: int = Query(20, ge=1, le=50)
):
    try:
        result = scraper.scrape(search_term=search_term, scrolls=scrolls, top_n=top_n)
        print(result, search_term)

        urls = result['top_images']

        print(json.dumps({'top_images': urls}, indent=4))
        return {'top_images': urls}
    except:
        return {'top_images': []}


def sample_from_distribution(distribution: dict, k: int = 1) -> list:
    items = list(distribution.keys())
    weights = list(distribution.values())
    return random.choices(items, weights=weights, k=k)


class Distribution(BaseModel):
    uploadedTotal: int
    tags: Dict[str, float]
    eras: Dict[str, float]


class DistributionUpdateRequest(BaseModel):
    current_distribution: Distribution
    accepted_images: List[str]
    rejected_images: List[str]


def image_to_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")  # or PNG if needed
    return buffer.getvalue()

def clean_distribution(distribution):
    return {
        "tags": {k: float(v) for k, v in distribution.get("tags", {}).items()},
        "eras": {k: float(v) for k, v in distribution.get("eras", {}).items()},
        "uploadedTotal": int(distribution.get("uploadedTotal", 0))
    }


@app.post("/update_distribution")
async def update_distribution(payload: DistributionUpdateRequest):
    current_distribution = payload.current_distribution
    accepted_images = payload.accepted_images
    rejected_images = payload.rejected_images

    print(f"Accepted images: {accepted_images}")
    print(f"Rejected images: {rejected_images}")

    # Load the tags from the current distribution
    current_tags = current_distribution.tags
    tag_keys = list(current_tags.keys())

    # Get the embeddings for the tags in the current distribution
    tag_embeddings = clip_encode_text(tag_keys)

    # Download and encode images (for both accepted and rejected)
    all_images = []
    for url in accepted_images + rejected_images:
        response = requests.get(url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert("RGB")
            all_images.append(image)

    # Encode the images using CLIP
    image_encodings = [
        await clip_encode_image(image_to_bytes(image)) for image in all_images
    ]

    # Initialize the new tag values as a copy of the current ones
    new_tag_values = defaultdict(float)
    total_images = len(accepted_images) + len(rejected_images)

    # Calculate cosine similarity and update the tag values based on the images
    for img_embedding in image_encodings:
        for tag, tag_embedding in zip(tag_keys, tag_embeddings):
            # Calculate cosine similarity between image and tag embedding
            similarity = cosine_similarity(img_embedding, tag_embedding)
            new_tag_values[tag] += similarity

    # Normalize the new tag values by averaging and renormalizing them
    new_tag_values_normalized = normalize_distribution(new_tag_values)

    # Now combine these values with the original distribution tags
    final_tag_distribution = defaultdict(float)
    for tag in current_tags:
        final_tag_distribution[tag] = current_tags.get(tag, 0.0) + new_tag_values_normalized.get(tag, 0.0)

    # Renormalize the final tag distribution
    final_tag_distribution_normalized = normalize_distribution(final_tag_distribution)

    # Keep the original era distribution as is
    new_era_distribution = current_distribution.eras

    distribution = {
        "tags": final_tag_distribution_normalized,
        "eras": new_era_distribution,  # Don't change the era distribution
        "uploadedTotal": current_distribution.uploadedTotal + len(accepted_images) + len(rejected_images),
    }
    cleaned_distribution = clean_distribution(distribution)
    # Return the updated distribution
    return cleaned_distribution


# Normalize a distribution (used to make sure tag values sum to 1)
def normalize_distribution(distribution: dict) -> dict:
    total_value = sum(distribution.values())
    if total_value == 0:
        return {key: 0.0 for key in distribution}
    return {key: value / total_value for key, value in distribution.items()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
