import requests
import torch
from transformers import PiperForTextToSpeech, PiperTokenizer

#  API endpoint
url = "http://localhost:1234/v1/chat/completions"

# Set up - Piper TTS and tokenizer
model_name = "piper-base"
tokenizer = PiperTokenizer.from_pretrained(model_name)
model = PiperForTextToSpeech.from_pretrained(model_name)


def send_to_tts(llm_response):
    # Preprocess the LLM response text
    input_text = llm_response["text"]
    inputs = tokenizer.encode_plus(
        input_text,
        return_tensors="pt",
        max_length=512,
        padding="max_length",
        truncation=True,
    )
    outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    audio = outputs["audio"]
    return audio

def get_llm_response():
    response = requests.post(url, json={"prompt": "Undefined"}) #Keyboard prompt
    response.raise_for_status()
    return response.json()


while True:
    llm_response = get_llm_response()
    audio = send_to_tts(llm_response)
    print("Audio synthesized!")