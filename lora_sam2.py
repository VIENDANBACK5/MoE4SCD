# lora_sam2.py
"""
LoRA wrapper for SAM2 image encoder.
Only add trainable params to attention layers (q_proj, v_proj).
"""
import torch.serialization
if not hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals = lambda *args, **kwargs: None

import torch
import torch.nn as nn

def apply_lora_to_sam2(sam2_model, rank: int = 4, alpha: float = 8.0):
    """
    Applies LoRA to the SAM2 image encoder.
    """
    try:
        from peft import LoraConfig, get_peft_model
        
        # SAM2 image encoder attention layers
        # target modules: q_proj, v_proj
        target_modules = ["q_proj", "v_proj"]
        
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        
        encoder = sam2_model.image_encoder
        encoder = get_peft_model(encoder, lora_config)
        sam2_model.image_encoder = encoder
        
        total = sum(p.numel() for p in sam2_model.parameters())
        trainable = sum(p.numel() for p in sam2_model.parameters() if p.requires_grad)
        print(f"LoRA applied (r={rank}):")
        print(f"  Total params:     {total:,}")
        print(f"  Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")
        print(f"  Target modules:   {target_modules}")
        
        return sam2_model
    except ImportError:
        print("[ERROR] peft not installed. Run: pip install peft")
        raise
    except AttributeError as e:
        print(f"[ERROR] SAM2 model structure not as expected: {e}")
        raise

def get_lora_lr_groups(model, lora_lr: float = 1e-4, base_lr: float = 1e-6):
    """
    Separate parameters into different learning rate groups.
    """
    lora_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(param)
        else:
            other_params.append(param)
            
    return [
        {"params": lora_params,  "lr": lora_lr,  "name": "lora"},
        {"params": other_params, "lr": base_lr,   "name": "other"},
    ]

if __name__ == "__main__":
    print("Testing LoRA application mock...")
    class MockSAM2:
        class MockEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)
            def forward(self, x):
                return self.q_proj(x) + self.v_proj(x)
        def __init__(self):
            self.image_encoder = self.MockEncoder()
        def parameters(self):
            return self.image_encoder.parameters()
        def named_parameters(self):
            return self.image_encoder.named_parameters()

    mock = MockSAM2()
    try:
        mock = apply_lora_to_sam2(mock, rank=4, alpha=8.0)
        print("✅ LoRA application mock successful!")
    except Exception as e:
        print(f"Mock test failed: {e}")
