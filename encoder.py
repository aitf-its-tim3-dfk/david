"""
Simplified ViFi-CLIP feature extractor loader.

Replaces the entire ViFi-CLIP repo + yacs config + mmcv dependency chain
with a single function: load_feature_extractor().

Architecture is kept compatible with ViFi-CLIP checkpoints so you can load
either base CLIP weights or finetuned ViFi-CLIP weights.
"""

import logging

import torch
import torch.nn as nn

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
logger = logging.getLogger(__name__)

# ── Default design details (matches the 16_16_vifi_clip.yaml config) ─────────
# These control the prompt-learning hooks baked into the architecture.
# With depth=0, no learnable prompts are active — it's basically vanilla CLIP.
DEFAULT_DESIGN = {
    "trainer": "ViFi_CLIP",
    "vision_depth": 0,
    "language_depth": 1,
    "vision_ctx": 0,
    "language_ctx": 0,
}


class TextEncoder(nn.Module):
    """Wraps CLIP's text transformer for standalone use."""

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class ViFiCLIP(nn.Module):
    """
    ViFi-CLIP wrapper around a base CLIP model.

    Provides:
      - .image_encoder  (the ViT visual backbone)
      - .text_encoder   (the text transformer)
      - .forward()      (full CLIP-style video classification — unused for feature extraction)
    """

    def __init__(self, clip_model, class_names, ctx_init="a photo of a"):
        super().__init__()
        dtype = clip_model.dtype

        # Build simple text embeddings (no prompt learning)
        prompts = [ctx_init.replace("_", " ") + " " + str(name) + "." for name in class_names]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("complete_text_embeddings", embedding)
        self.tokenized_prompts = tokenized_prompts

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = dtype

    def forward(self, image):
        """Full ViFi-CLIP forward (video -> class logits). Not used for feature extraction."""
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        prompts = self.complete_text_embeddings

        b, t, c, h, w = image.size()
        image = image.reshape(-1, c, h, w)
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features.view(b, t, -1).mean(dim=1)

        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return logit_scale * image_features @ text_features.t()


def load_feature_extractor(
    arch="ViT-B/16",
    class_names=("true", "false"),
    checkpoint_path=None,
    design_details=None,
):
    """
    Load a ViFi-CLIP model as a frozen feature extractor.

    Parameters
    ----------
    arch : str
        CLIP backbone name (e.g. "ViT-B/16").
    class_names : tuple[str]
        Class label names for the text encoder.
    checkpoint_path : str, optional
        Path to a ViFi-CLIP finetuned checkpoint (.pth).
        If None, uses the base OpenAI CLIP weights.
    design_details : dict, optional
        Architecture config for prompt hooks. Defaults to DEFAULT_DESIGN
        (prompt learning disabled).

    Returns
    -------
    ViFiCLIP
        Model on CPU in float32, with all gradients enabled.
        Call .cuda() to move to GPU.
    """
    if design_details is None:
        design_details = DEFAULT_DESIGN

    # 1. Download and build the base CLIP architecture
    logger.info(f"Loading CLIP backbone: {arch}")
    url = clip._MODELS[arch]
    model_path = clip._download(url)

    try:
        jit_model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = jit_model.state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    clip_model = clip.build_model(state_dict, design_details)

    # 2. Wrap in ViFiCLIP
    model = ViFiCLIP(clip_model, class_names)

    # 3. Optionally load finetuned weights
    if checkpoint_path is not None:
        logger.info(f"Loading finetuned weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        # Handle both raw state_dicts and checkpoints with a 'state_dict' key
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")

    # 4. Enable all gradients (matches original ViFi-CLIP "both" mode)
    for param in model.parameters():
        param.requires_grad_(True)

    model.float()
    logger.info("Feature extractor ready.")
    return model
