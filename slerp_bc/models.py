"""Vision-language models used in the paper (loaded with OpenCLIP)."""

import open_clip

# key: (OpenCLIP architecture, pretrained tag)
MODELS = {
    "b32":     ("ViT-B-32", "openai"),
    "l14":     ("ViT-L-14", "openai"),
    "h14":     ("ViT-H-14", "laion2b_s32b_b79k"),
    "siglip1": ("ViT-SO400M-14-SigLIP", "webli"),
    "siglip2": ("ViT-SO400M-14-SigLIP2", "webli"),
}


def load_model(key, device):
    """Return (model, image preprocessing, tokenizer) for a model key."""
    arch, pretrained = MODELS[key]
    # OpenAI CLIP weights were trained with QuickGELU.
    model, _, preprocess = open_clip.create_model_and_transforms(
        arch, pretrained=pretrained, device=device, force_quick_gelu=pretrained == "openai")
    return model.eval(), preprocess, open_clip.get_tokenizer(arch)
