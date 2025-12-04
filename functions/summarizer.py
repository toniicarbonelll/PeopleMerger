from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import numpy as np
import re
from dataclasses import dataclass
import time
import pickle
import numpy as np
import torch


def load_tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model


@dataclass
class SummParams:
    text: str
    tokenizer: AutoTokenizer
    model: AutoModelForSeq2SeqLM
    num_beams: int = 6
    max_input_length: int = 1024
    min_output_length: int = 20
    max_output_length: int = 200
    device: torch.device = torch.device("cuda")


def summarize_text(summparams: SummParams) -> str:
    """Summarizes text

    Args:
        summparams (SummParams): Parameters for text summarization (dataclass)

    Returns:
        str: Summarized text
    """
    inputs = summparams.tokenizer([summparams.text], max_length=summparams.max_input_length, return_tensors='pt').to(summparams.device)
    summary_ids = summparams.model.generate(inputs['input_ids'], num_beams=summparams.num_beams, min_length=summparams.min_output_length, max_length=summparams.max_output_length, early_stopping=True)
    return (summparams.tokenizer.decode(summary_ids[0], skip_special_tokens=True, truncation=True))


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


def generate_prompt(images_dict: dict)->str:
    """Generates a prompt from the images dictionary by combining three random elements 

    Args:
        images_dict (dict): Images dictionary

    Returns:
        str: Generated prompt
    """
    #choosing three random elements
    keys = list(images_dict.keys()) 
    chosen_keys = [str(string) for string in np.random.choice(keys, 3, replace=False)] 
    print(chosen_keys)

    #building the prompt
    prompt = images_dict[chosen_keys[0]]['person']
    prompt += images_dict[chosen_keys[1]]['clothes']
    prompt += images_dict[chosen_keys[2]]['scenario']
    print(f"Final prompt (length: {len(prompt.split(" "))}): {prompt}")
    return prompt


def summarize_prompt(prompt: str, tokenizer, model, device) ->str:
    """Summarizes the generated prompt using BART-LARGE-CNN three times (sequentially), so that it fits into 77 tokens (approx 55 words).

    Args:
        prompt (str): Generated prompt.
        tokenizer: Tokenizer.
        model: Model.
        device: Device.

    Returns:
        str: Prompt summarized into 77 tokens (55 words)
    """

    # First summarization 
    summparams = SummParams(
        text = prompt,
        tokenizer = tokenizer,
        model = model,
        num_beams = 4,
        max_input_length = 1024,
        min_output_length = 90,
        max_output_length= 100,
        device = device
    )
    summarized_text_1 = summarize_text(summparams)
    print(f"Summary 1 (length: {len(summarized_text_1.split(" "))}): {summarized_text_1}")

    # Second summarization
    summparams.text = summarized_text_1
    summparams.num_beams = 2
    summparams.max_input_length = 100
    summparams.min_output_length = 80
    summparams.max_output_length = 85
    summarized_text_2 = summarize_text(summparams)
    print(f"Summary 2 (length: {len(summarized_text_2.split(" "))}): {summarized_text_2}")



    # Third summarization
    summparams.text = summarized_text_2
    summparams.num_beams = 1
    summparams.max_input_length = 100
    summparams.min_output_length = 60
    summparams.max_output_length = 65
    summarized_text_3 = summarize_text(summparams)
    print(f"Summary 3 (length: {len(summarized_text_3.split(" "))}): {summarized_text_3}")
    return summarized_text_3