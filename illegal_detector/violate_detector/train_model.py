from peft import get_peft_model, LoraConfig
from transformers import CLIPModel, CLIPProcessor
import torch

# 1. Load CLIP và áp dụng LoRA
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, config)

# 2. Dataset xử lý (Sử dụng CLIPProcessor)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def train_step(image, text_prompt):
    # Process inputs
    inputs = processor(text=text_prompt, images=image, return_tensors="pt", padding=True)
    
    # Forward
    outputs = model(**inputs)
    
    # Loss: Đây là phần quan trọng nhất
    # CLIP sử dụng Contrastive Loss (tự động có sẵn trong model)
    loss = outputs.loss 
    
    loss.backward()
    return loss.item()

model.save_pretrained("./my_violation_lora_adapter")