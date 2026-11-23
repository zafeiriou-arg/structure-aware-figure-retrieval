import re

import fitz
import numpy as np
import pandas as pd
from PIL import Image

from figret.geometry import frac_of_a, keep_larger

KINDS = {"Picture": "picture", "Caption": "caption", "Text": "text", "List-item": "text", "Section-header": "header", "Title": "title"}


def render_page(doc, i, dpi):
    pix = doc[i].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def detect(model, img, conf=0.25):
    res = model.predict(img, imgsz=1024, conf=conf, verbose=False)[0]
    out = []
    for box, cls, score in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
        kind = KINDS.get(model.names[int(cls)])
        if kind:
            out.append((kind, tuple(box), float(score)))
    return out


def page_words(page, zoom):
    return [((w[0] * zoom, w[1] * zoom, w[2] * zoom, w[3] * zoom), w[4]) for w in page.get_text("words")]


def words_in(box, words):
    inside = [(b, t) for b, t in words if box[0] <= (b[0] + b[2]) / 2 <= box[2] and box[1] <= (b[1] + b[3]) / 2 <= box[3]]
    inside.sort(key=lambda w: (round(w[0][1] / 8), w[0][0]))
    out = []
    for k, (b, t) in enumerate(inside):
        line_end = k + 1 == len(inside) or round(inside[k + 1][0][1] / 8) != round(b[1] / 8)
        out.append(t[:-1] if t.endswith("-") and line_end and k + 1 < len(inside) else t + " ")
    return clean_text("".join(out))


DECOR = re.compile(r"[^\w\s.,;:()\[\]\-–—'’‘\"“”/%°&+=<>±×µ?!]")


def clean_text(text):
    return re.sub(r"\s+", " ", DECOR.sub(" ", text)).strip()


FIG_PREFIX = re.compile(r"^\W*(fig(ure|ura)?s?\.?|plate|l[aá]mina|prancha)\s*(?:[A-Za-z]{1,4}-)?[\d.]+[a-z]?\b\S*\s*([A-ZÀ-Ý(\[:.|–—-]|$)", re.I)


def fix_captions(rows, gap=20):
    rows = [dict(r) for r in rows]
    for r in rows:
        m = FIG_PREFIX.match(r["text"])
        if r["kind"] in ("text", "header") and m and not m.group(3).islower():
            r["kind"] = "caption"
    out = []
    for r in sorted(rows, key=lambda r: (r["y0"], r["x0"])):
        below = None
        for c in reversed(out):
            if c["kind"] == "caption" and len(c["text"].split()) <= 15 and -10 <= r["y0"] - c["y1"] <= gap:
                overlap = min(c["x1"], r["x1"]) - max(c["x0"], r["x0"])
                if overlap > 0.5 * min(c["x1"] - c["x0"], r["x1"] - r["x0"]):
                    below = c
                    break
        if r["kind"] == "text" and below is not None and not below.get("merged"):
            below.update(text=below["text"] + " " + r["text"], merged=True,
                         x0=min(below["x0"], r["x0"]), x1=max(below["x1"], r["x1"]), y1=r["y1"])
            continue
        out.append(r)
    for r in out:
        r.pop("merged", None)
    return out


VLM_OCR_PROMPT = ("Transcribe all text in this image exactly as printed, in reading order, keeping the spaces "
                  "between words. Answer with the text only.")


class VlmOcr:
    def __init__(self, vlm, max_new_tokens=400):
        self.vlm, self.max_new_tokens = vlm, max_new_tokens

    def __call__(self, img, box):
        crop = img.crop(tuple(int(v) for v in box))
        return clean_text(self.vlm.ask([crop], VLM_OCR_PROMPT, max_new_tokens=self.max_new_tokens))


def ocr_region(engine, img, box):
    crop = np.array(img.crop(tuple(int(v) for v in box)))
    result, _ = engine(crop)
    if not result:
        return ""
    result.sort(key=lambda r: (round(r[0][0][1] / 10), r[0][0][0]))
    return clean_text(" ".join(r[1] for r in result))


def parse_page(model, ocr, doc, atlas, i, dpi, crops_dir):
    img = render_page(doc, i, dpi)
    regions = detect(model, img)
    words = page_words(doc[i], dpi / 72)
    use_ocr = atlas.ocr or sum(len(t) for _, t in words) < 20

    pics = [r for r in regions if r[0] == "picture"]
    pics = [pics[k] for k in keep_larger([p[1] for p in pics])]
    texts = [r for r in regions if r[0] != "picture" and all(frac_of_a(r[1], p[1]) <= 0.5 for p in pics)]

    rows = []
    for n, (kind, box, score) in enumerate(sorted(pics, key=lambda r: (r[1][1], r[1][0]))):
        eid = f"X:{atlas.id}:{i}:{n}"
        path = crops_dir / f"{eid.replace(':', '_')}.png"
        img.crop(tuple(int(v) for v in box)).save(path)
        rows.append(dict(element_id=eid, kind=kind, box=box, text="", crop_path=str(path), det_score=score))
    for r in rows:
        r["x0"], r["y0"], r["x1"], r["y1"] = r.pop("box")
    trows = []
    for kind, box, score in texts:
        text = ocr(img, box) if use_ocr else words_in(box, words)
        if kind == "text" and len(text.split()) < 3:
            continue
        trows.append(dict(kind=kind, x0=box[0], y0=box[1], x1=box[2], y1=box[3], text=text, crop_path="", det_score=score))
    for n, r in enumerate(fix_captions(trows)):
        rows.append(dict(element_id=f"E:{atlas.id}:{i}:{n}", **r))
    for r in rows:
        r.update(atlas=atlas.id, page=i, ocr=use_ocr, page_w=img.width, page_h=img.height)
    return img, rows


def parse_atlas(cfg, atlas, pages=None):
    from rapidocr_onnxruntime import RapidOCR
    from ultralytics import YOLO

    model = YOLO(str(cfg.root / cfg.models["layout"]))
    if atlas.ocr == "vlm":
        from figret import vlm

        ocr = VlmOcr(vlm.load(cfg.models["vlm"]))
    else:
        engine = RapidOCR()
        ocr = lambda img, box: ocr_region(engine, img, box)
    doc = fitz.open(cfg.pdf(atlas))
    crops_dir = cfg.run_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in pages if pages is not None else range(len(doc)):
        rows += parse_page(model, ocr, doc, atlas, i, cfg.dpi, crops_dir)[1]
    cols = ["element_id", "atlas", "page", "kind", "x0", "y0", "x1", "y1", "text", "crop_path", "det_score", "ocr", "page_w", "page_h"]
    return drop_running_text(pd.DataFrame(rows, columns=cols))


def drop_running_text(df, min_pages=5):
    text = df.text.str.lower().str.replace(r"\d+", "", regex=True).str.strip()
    counts = df[df.kind != "picture"].assign(t=text).groupby("t").page.nunique()
    running = set(counts[counts >= min_pages].index) - {""}
    return df[~((df.kind != "picture") & text.isin(running))].reset_index(drop=True)


def load_elements(cfg):
    ids = {a.id for a in cfg.atlases}
    paths = [p for p in sorted((cfg.run_dir / "elements").glob("*.parquet")) if p.stem in ids]
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
