from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from .env into os.environ

from huggingface_hub import login

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

from fastapi import FastAPI, Request
from pydantic import BaseModel
from huggingface_hub import from_pretrained_keras
import tensorflow as tf

app = FastAPI()


model_id = "Hitarth28/finetuned-gemma"
gemma_lm = from_pretrained_keras(model_id)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return {"error": "Prompt is empty."}

        # Generate from model
        result = gemma_lm.generate({"prompts": [prompt]})
        full_output = result[0]

        # Strip prompt from the generated output
        response_only = full_output[len(prompt):].strip()

        return {
            "prompt": prompt,
            "response": response_only
        }

    except Exception as e:
        return {"error": str(e)}

