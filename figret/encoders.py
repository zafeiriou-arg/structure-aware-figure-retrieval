import numpy as np
import torch
from PIL import Image


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)


def features(out):
    return getattr(out, "pooler_output", out)


class TextEncoder:
    def __init__(self, name, device="cuda"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name, device=device)
        self.query_prompt = "query" if "query" in (self.model.prompts or {}) else None

    def embed(self, texts, query=False, batch_size=64):
        kw = {"prompt_name": self.query_prompt} if query and self.query_prompt else {}
        return norm(self.model.encode(list(texts), batch_size=batch_size, **kw))


class OpenClipEncoder:
    def __init__(self, spec, device="cuda"):
        import open_clip

        if spec.startswith("hf-hub:"):
            self.model, self.preprocess = open_clip.create_model_from_pretrained(spec)
            self.tokenizer = open_clip.get_tokenizer(spec)
        else:
            arch, tag = spec.split("/")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=tag)
            self.tokenizer = open_clip.get_tokenizer(arch)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def embed_images(self, paths, batch_size=32):
        out = []
        for i in range(0, len(paths), batch_size):
            x = torch.stack([self.preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]])
            out.append(self.model.encode_image(x.to(self.device)).float().cpu())
        return norm(torch.cat(out))

    @torch.inference_mode()
    def embed_texts(self, texts, batch_size=64):
        texts = list(texts)
        out = []
        for i in range(0, len(texts), batch_size):
            out.append(self.model.encode_text(self.tokenizer(texts[i:i + batch_size]).to(self.device)).float().cpu())
        return norm(torch.cat(out))


class SiglipEncoder:
    def __init__(self, name, device="cuda", context=64):
        from transformers import AutoModel, AutoProcessor

        self.name = name
        self.model = AutoModel.from_pretrained(name).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(name)
        self.device = device
        self.context = context

    @torch.inference_mode()
    def embed_images(self, paths, batch_size=16):
        out = []
        for i in range(0, len(paths), batch_size):
            ims = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
            x = self.processor(images=ims, return_tensors="pt").to(self.device)
            out.append(features(self.model.get_image_features(**x)).float().cpu())
        return norm(torch.cat(out))

    @torch.inference_mode()
    def embed_texts(self, texts, batch_size=64):
        texts = [t.lower() for t in texts] if "siglip2" in self.name else list(texts)
        out = []
        for i in range(0, len(texts), batch_size):
            x = self.processor(text=texts[i:i + batch_size], padding="max_length", truncation=True,
                               max_length=self.context, return_tensors="pt").to(self.device)
            out.append(features(self.model.get_text_features(**x)).float().cpu())
        return norm(torch.cat(out))


def image_text(spec, device="cuda"):
    return SiglipEncoder(spec, device) if "siglip" in spec else OpenClipEncoder(spec, device)


def text(name, device="cuda"):
    return TextEncoder(name, device)
