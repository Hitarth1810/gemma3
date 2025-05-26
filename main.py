import os
from huggingface_hub import login, from_pretrained_keras
from fastapi import FastAPI, Request
from pydantic import BaseModel
import tensorflow as tf

# Log in to Hugging Face
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# Load the model
model_id = "Hitarth28/finetuned-gemma"
gemma_lm = from_pretrained_keras(model_id)

# Set up FastAPI
app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return {"error": "Prompt is empty."}

        result = gemma_lm.generate({"prompts": [prompt]})
        full_output = result[0]
        response_only = full_output[len(prompt):].strip()

        return {
            "prompt": prompt,
            "response": response_only
        }
    except Exception as e:
        return {"error": str(e)}
