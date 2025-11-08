import torch
from typing import Dict

def monitor_ebc_ratio(model, hook_handles=None):
    """Monitor grad/act ratio per layer"""
    ratios = {}
    for name, param in model.named_parameters():
        if param.grad is None: continue
        mod_name = name.rsplit('.', 1)[0]
        mod = dict(model.named_modules())[mod_name]
        if hasattr(mod, '_ebc_act'):
            act_norm = mod._ebc_act.norm()
            grad_norm = param.grad.norm()
            ratios[name] = grad_norm / (act_norm + 1e-8)
    return ratios
