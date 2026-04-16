import os
import torch
import warnings
from transformers import AutoProcessor
from torch.utils.data import DataLoader

from reinspection_internvl3.config import ReInspectionConfig
from reinspection_internvl3.modeling import load_model, InternVL3WithReInspection
from reinspection_internvl3.data.spatial_vqa import spatial_vqa_dataset
from reinspection_internvl3._train_common import _collate_fn as collate_fn

warnings.filterwarnings("ignore")

def debug_nan():
    print("Loading config and processor...")
    config = ReInspectionConfig(bf16=True, max_pixels=256*256)
    
    # Needs to be bfloat16 to replicate training exactly
    dtype = torch.bfloat16
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, dtype: {dtype}")
    
    # Load model
    print("Loading model...")
    model = load_model(config, device_map=device)
    model.to(dtype)
    model.train()
    
    # Load small subset of data
    print("Loading data...")
    processor = AutoProcessor.from_pretrained(config.model_name_or_path)
    
    # Just need one valid sample
    from reinspection_internvl3.data.spatial_vqa import VSRDataset
    dataset = VSRDataset("/data/datasets/vision_text", processor, "train")
    
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    batch = next(iter(loader))
    
    # Move to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # Debug hook
    print("\n--- Running Forward Pass with NaN Tracking ---")
    
    # We will hook into the inner methods by modifying them temporarily or just running them
    # Let's extract inputs
    input_ids = batch.get("input_ids")
    attention_mask = batch.get("attention_mask")
    pixel_values = batch.get("pixel_values")
    image_flags = batch.get("image_flags")
    labels = batch.get("labels")
    
    print(f"input_ids: {input_ids.shape}")
    if pixel_values is not None:
        print(f"pixel_values: {pixel_values.shape}, image_flags: {image_flags.shape if image_flags is not None else None}")
    
    with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=dtype):
        # 1. Inputs Embeds
        inputs_embeds = model.base_model.model.get_input_embeddings()(input_ids)
        print(f"Initial inputs_embeds NaN: {torch.isnan(inputs_embeds).any().item()}")
        
        # 2. Vision Encoding
        prepared = model._prepare_reinspection_inputs(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_flags=image_flags,
            need_weights=False
        )
        
        print(f"After _prepare_reinspection_inputs embeds NaN: {torch.isnan(prepared['inputs_embeds']).any().item()}")
        print(f"R tokens (A_task) NaN: {torch.isnan(prepared['A_task']).any().item() if prepared.get('A_task') is not None else False}")
        
        # Let's run the language model
        outputs = model.base_model.model.language_model(
            inputs_embeds=prepared["inputs_embeds"],
            attention_mask=prepared["attention_mask"],
            output_hidden_states=False,
            use_cache=False,
        )
        
        hidden_states = outputs.last_hidden_state
        print(f"LLM output hidden_states NaN: {torch.isnan(hidden_states).any().item()}")
        
        logits = model.base_model.lm_head(hidden_states)
        print(f"Logits NaN: {torch.isnan(logits).any().item()}")
        
        loss = model.base_model.loss_function(
            logits=logits,
            labels=prepared["labels"],
            vocab_size=model.base_model.config.text_config.vocab_size,
        )
        print(f"Loss NaN: {torch.isnan(loss).any().item()} (value: {loss.item()})")
        
if __name__ == "__main__":
    debug_nan()
