#import openai
import time, sys

import torch
from transformers import pipeline


""" def openai_completion(prompt, engine="gpt-3.5-turbo", max_tokens=700, temperature=0):
    client = openai.OpenAI()
    
    resp =  client.chat.completions.create(
        model=engine,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["\n\n", "<|endoftext|>"]
        )
    
    return resp.choices[0].message.content """

def openchat_completion(prompt, model="openchat/openchat_3.5",max_tokens=700, temperature=0):
    if(torch.cuda.is_available()):
        device = 'cuda'
    else:
        device = 'cpu'

    messages = [
        {"role": "user", "content": prompt},
    ]

    pipe = pipeline("text-generation", 
                    model="openchat/openchat_3.5", 
                    device=device)
    
    response = pipe(messages,
                    #max_new_tokens=max_tokens, 
                    temperature=temperature, 
                    stop_sequence=["\n\n", "<|endoftext|>"])
    
    #print(response[0]['generated_text'][1])
    #input()
    
    return response[0]['generated_text']
