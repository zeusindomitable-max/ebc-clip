import torch
from typing import Callable, Dict, Any, Optional
from torch import nn

def energy_budget_clip(
    model: nn.Module,
    ratio: float = 0.1,
    hook_handles: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> Callable[[], None]:
    """
    Energy-Budget Gradient Clipping.
    
    Ensures: ||∇θL|| <= ratio * ||activation(θ)||
    
    Args:
        model: PyTorch model
        ratio: 0.1 (universal default)
        hook_handles: Reuse existing hooks
        verbose: Print clipping stats
    
    Returns:
        clip_fn: Call after loss.backward()
    """
    if hook_handles is None:
        hook_handles = {}

    def forward_hook(module, input, output):
        module._ebc_act = output.detach()

    # Register hooks once
    for name, module in model.named_modules():
        if name not in hook_handles and hasattr(module, 'weight'):
            hook_handles[name] = module.register_forward_hook(forward_hook)

    def clip_fn():
        clipped = 0
        total = 0
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            total += 1
            mod_name = name.rsplit('.', 1)[0]
            mod = dict(model.named_modules())[mod_name]
            if hasattr(mod, '_ebc_act'):
                act_norm = mod._ebc_act.norm()
                threshold = ratio * act_norm
                grad_norm = param.grad.norm()
                if grad_norm > threshold:
                    param.grad.data *= (threshold / grad_norm)
                    clipped += 1
        if verbose and clipped > 0:
            print(f"[EBC] Clipped {clipped}/{total} parameters")

    return clip_fn
