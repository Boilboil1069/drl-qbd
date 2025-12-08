import os
import torch

# Allow overriding device via environment, e.g. to force a specific GPU or CPU
# DRL_QBD_DEVICE can be:
#   - 'cpu'          -> force CPU
#   - 'cuda'         -> default CUDA device (cuda:0)
#   - 'cuda:0', 'cuda:1', ... -> specific GPU index
#   - 'mps'          -> Apple Silicon GPU via Metal (if available)
# If not set, fall back to automatic selection.
_env_device = os.environ.get("DRL_QBD_DEVICE", "").strip().lower()

if _env_device:
    if _env_device == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            print("[WARN] DRL_QBD_DEVICE=mps but MPS is not available, falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device(_env_device)
else:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


def set_seed(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
