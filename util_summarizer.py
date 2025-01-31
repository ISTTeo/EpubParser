from typing import Dict, List, Tuple, Optional, Any
from openai import OpenAI
import tiktoken
import json, os, pickle, requests


def summarize_text_OPENAI(messages, openai_key, model):
    client = OpenAI(api_key=openai_key)

    response = client.chat.completions.create(
        model=model, 
        messages = messages,
    )
    summary = response.choices[0].message.content.strip()
    return summary, messages

def summarize_text_LMSTUDIO(messages, tries=5):
    
    payload = {
        "model": "phi-4@q6_k", 
        "messages": messages,
    }
    
    

    while tries > 0:
        try:
            response = requests.post(
                "http://localhost:24236/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=15
            )
            if response.status_code == 200:
                result = response.json()
                summary = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                return summary, messages
            else:
                tries -=1 
        except:
            tries -=1

    raise Exception(f"Tries ran out --> {messages}")



def summarize_text(messages, openai_key=None, model="gpt-4o-mini"):
    if openai_key is not None:
        return summarize_text_OPENAI(messages, openai_key, model)
        
    return summarize_text_LMSTUDIO(messages)

def deep_copy_dict(dictionary):
    # Base case - if input is not a dictionary
    if not isinstance(dictionary, dict):
        # If it's a list, deep copy each element
        if isinstance(dictionary, list):
            return [deep_copy_dict(item) for item in dictionary]
        # If it's any other type, return as is (assuming immutable)
        return dictionary
    
    # Create new dictionary for the copy
    copy = {}
    
    # Iterate through key-value pairs
    for key, value in dictionary.items():
        # Recursively copy nested dictionaries/lists
        copy[key] = deep_copy_dict(value)
        
    return copy




