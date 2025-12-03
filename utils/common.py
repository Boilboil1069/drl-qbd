import os
import torch

# Allow overriding device via environment, e.g. to force a specific GPU or CPU
# DRL_QBD_DEVICE can be:
#   - 'cpu'          -> force CPU
#   - 'cuda'         -> default CUDA device (cuda:0)
#   - 'cuda:0', 'cuda:1', ... -> specific GPU index
# If not set, fall back to automatic selection.
_env_device = os.environ.get("DRL_QBD_DEVICE", "").strip()

if _env_device:
    device = torch.device(_env_device)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
