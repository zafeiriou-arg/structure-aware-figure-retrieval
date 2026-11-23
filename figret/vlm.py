import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def shrink(img, max_side):
    img = img.convert("RGB")
    scale = max_side / max(img.size)
    if scale < 1:
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    return img


class VLM:
    def __init__(self, name, max_side=1024):
        self.name = name
        self.max_side = max_side
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()

    @torch.inference_mode()
    def ask(self, images, prompt, max_new_tokens=256):
        content = [{"type": "image", "image": shrink(im, self.max_side)} for im in images]
        messages = [{"role": "user", "content": content + [{"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def load(name, max_side=1024):
    return VLM(name, max_side)
