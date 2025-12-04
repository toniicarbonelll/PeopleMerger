from fastapi import FastAPI, Request
import sys
import os 
sys.path.append(os.path.join(os.getcwd(), '../'))
from credentials import Credentials
credentials = Credentials()
os.environ["http_proxy"] = credentials.http_proxy
os.environ["https_proxy"] = credentials.https_proxy
import numpy as np
import pickle
import torch
from PIL import Image
import time
from pydantic import BaseModel
from typing import List
from fastapi.responses import JSONResponse
import json

app = FastAPI()

############################################################
# 1. Loading GEMMA 
# Load model on GPU if available
print("Loading GEMMA...")
print("Logging to Hugging Face...")
from huggingface_hub import login
login(token=credentials.hf_token)

print("Importing transformer models...")
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print("Loading the models...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,             # or False if you want full precision
    bnb_4bit_quant_type="nf4",     # 'nf4' or 'fp4'
    bnb_4bit_compute_dtype=torch.float16,  # can also try torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

print("Moving models to GPU...")
device = torch.device("cuda")
model = model.to(device)


############################################################
# 2. Loading FLUX (glups) 
print("Loading Flux...")
from diffusers import FluxPipeline
from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision

def clean_cache():
    print("Cleaning cache...")
    for i in range(5):
        torch.cuda.empty_cache()
        time.sleep(4)

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
is_loaded_flux = False
print("Loading Transformer...")
for i in range(10):
    print("Try number: ", i, " ...")
    if is_loaded_flux:
        break
    try:
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            f"nunchaku-tech/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
        )
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
        ).to("cuda")
    except:
        torch.cuda.empty_cache()
        time.sleep(3)
    else:
        is_loaded_flux = True




######################################################################
# Defining functions
import base64
def extract_base64(image_file):
    with open(image_file, "rb") as img_file:
        base64_string = base64.b64encode(img_file.read()).decode('utf-8')
    return base64_string

def load_images_dict(images_dict_path:str)->dict:
    """Loads images dictionary from the path

    Args:
        images_dict_path (str): Path to images dictionary

    Returns:
        dict: Images dictionary
    """
    with open(images_dict_path, 'rb') as fp:
        images_dict = pickle.load(fp)
    return images_dict


#####################################################################
# Get fotos
from pydantic import BaseModel
class Foto(BaseModel):
    name: str
    person_description: str
    clothes_description: str
    scenario_description: str
    content: str
@app.get("/gemma_flux/get_fotos")
async def get_fotos():
    images = [img for img in os.listdir('../data') if len(img.split("."))>1 and img.split(".")[1] in ['png','jpg']]
    images_dict = load_images_dict('../notebooks/images_dict.pkl')

    fotos = []
    for image in images:
        image_info = images_dict[image]
        base64_encoding = extract_base64('../data/' + image)
        fotos.append({
            'name': image,
            'person_description': image_info['person'],
            'clothes_description': image_info['clothes'],
            'scenario_description': image_info['scenario'],
            'content': base64_encoding
        })
    
    return JSONResponse(content=fotos)

######################################################################
# SUMMARIZE
@app.post("/gemma_flux/summarize")
async def summarize(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    # customizing prompt
    prompt = 'Summarize this text into about 55 words: "' + prompt + '"'
    messages = [
        {"role": "user", "content": prompt},
    ]
    # tokenizing inputs
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    # generating outputs
    outputs = model.generate(**inputs, max_new_tokens=100)
    summary = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return {"summary": summary}

######################################################################
# # GENERATE_FOTO
@app.post("/gemma_flux/generate_image")
async def generate_image(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    # Cleaning cache
    clean_cache()
    # Generating image
    image = pipeline(
        prompt,
        num_inference_steps=20, #typical number of inference steps for this model
        guidance_scale=3.5,
        height=256,
        width=256
        ).images[0]
    # returning image on base64
    image.save("./generated_image.png")
    base64_encoding = extract_base64("./generated_image.png")
    return {"image": base64_encoding}