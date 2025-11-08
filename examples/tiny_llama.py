# TinyLlama demo — runs in 5 minutes
from transformers import AutoModelForCausalLM, AutoTokenizer
from ebc_clip import energy_budget_clip
import torch

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
clip_fn = energy_budget_clip(model, ratio=0.1)

text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")

for _ in range(10):
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()
    clip_fn()
    torch.optim.AdamW(model.parameters(), lr=1e-4).step()
    model.zero_grad()
    print(f"Loss: {loss.item():.3f}")
