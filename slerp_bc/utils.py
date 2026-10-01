import torch


def get_device():
    """Return "cuda" if PyTorch was built with CUDA and a GPU is usable, otherwise "cpu"."""
    if torch.version.cuda is None:
        print("PyTorch was installed without CUDA support: running on CPU.")
        return "cpu"
    if not torch.cuda.is_available():
        print(f"PyTorch was built with CUDA {torch.version.cuda}, but no usable GPU was found "
              "(no GPU visible, or NVIDIA driver too old for this build): running on CPU.")
        return "cpu"
    print(f"Running on {torch.cuda.get_device_name()} (CUDA {torch.version.cuda}).")
    return "cuda"
