import json
import re

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from figret.geometry import edge_distance, frac_of_smaller, nearest_caption

PROMPT = (
    "This is a schematic of a book page. The RED box is a picture. BLUE boxes are other pictures. "
    "GREEN boxes are text blocks, each labelled with a number. Which green box is the caption of the RED picture? "
    "A caption usually sits directly below, above or beside its picture, and one caption can describe several pictures. "
    "If none of the green boxes is its caption, answer null. "
    'Answer with JSON only: {"caption_id": <number or null>, "confidence": <0 to 1>}'
)


def box(r):
    return (r.x0, r.y0, r.x1, r.y1)


def candidates(page, near_px=60):
    pics = [box(p) for p in page[page.kind == "picture"].itertuples()]
    rows = page[page.kind.isin(["caption", "text"])].sort_values("kind")
    keep, kept_boxes = [], []
    for r in rows.itertuples():
        b = box(r)
        if any(frac_of_smaller(b, k) > 0.8 for k in kept_boxes):
            continue
        if r.kind == "caption" or any(edge_distance(b, p) <= near_px for p in pics):
            keep.append(r.Index)
            kept_boxes.append(b)
    return page.loc[keep]


def parse_choice(answer, ids):
    m = re.search(r"\{.*?\}", answer, re.S)
    try:
        data = json.loads(m.group(0))
        cid, conf = data.get("caption_id"), float(data.get("confidence", 0))
        if cid is None:
            return None, conf
        cid = int(cid)
    except (AttributeError, ValueError, TypeError):
        return None, 0.0
    return (cid, conf) if cid in ids else (None, 0.0)


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def label(draw, b, text, size):
    draw.text((b[0] + 4, b[1] + 2), text, fill="white", font=font(size), stroke_width=3, stroke_fill="black")


def synthetic_page(size, target, others, cands):
    img = Image.new("RGB", size, "black")
    d = ImageDraw.Draw(img)
    for b in others:
        d.rectangle(b, fill=(0, 0, 255))
    d.rectangle(target, fill=(255, 0, 0))
    for n, b in cands:
        d.rectangle(b, fill=(0, 200, 0))
        label(d, b, str(n), max(18, size[0] // 40))
    return img


def pair_page(vlm, page):
    pics = page[page.kind == "picture"]
    cands = candidates(page)
    numbered = list(enumerate(cands.itertuples(), start=1))
    cand_boxes = [box(c) for c in cands.itertuples()]
    rows = []
    for p in pics.itertuples():
        rule = nearest_caption(box(p), cand_boxes)
        row = dict(panel_id=p.element_id, rule_element=cands.element_id.iloc[rule] if rule is not None else None,
                   caption_element=None, confidence=0.0, answer="")
        if len(cands) == 1 and len(pics) == 1:
            row.update(caption_element=cands.element_id.iloc[0], confidence=1.0, answer="single")
        elif len(cands):
            others = [box(o) for o in pics.itertuples() if o.element_id != p.element_id]
            boxes = [(n, box(c)) for n, c in numbered]
            img = synthetic_page((int(p.page_w), int(p.page_h)), box(p), others, boxes)
            answer = vlm.ask([img], PROMPT, max_new_tokens=40)
            n, conf = parse_choice(answer, [n for n, _ in numbered])
            row.update(caption_element=cands.element_id.iloc[n - 1] if n else None, confidence=conf, answer=answer)
        rows.append(row)
    return rows


def run_pairing(vlm, elements, pages):
    rows = []
    for a, pg in pages:
        page = elements[(elements.atlas == a) & (elements.page == pg)]
        page_rows = pair_page(vlm, page)
        for r in page_rows:
            r.update(atlas=a, page=pg)
        rows += page_rows
    return pd.DataFrame(rows)


def group_panels(pairs):
    key = pairs.caption_element.where(pairs.caption_element.notna(), "solo:" + pairs.panel_id)
    return pairs.assign(key=key).groupby("key", sort=False).agg(
        panel_ids=("panel_id", list), caption_element=("caption_element", "first")).reset_index(drop=True)


def add_context(figs, elements, chapters):
    from figret.chapters import chapter_of

    by_id = elements.set_index("element_id")
    headers = elements[elements.kind == "header"].sort_values(["atlas", "page", "y0"])
    texts = elements[elements.kind == "text"]
    out = []
    for f in figs.itertuples():
        p = by_id.loc[f.panel_ids[0]]
        ch = chapter_of(chapters, p.atlas, p.page)
        h = headers[(headers.atlas == p.atlas) & ((headers.page < p.page) | ((headers.page == p.page) & (headers.y0 < p.y0)))]
        if ch is not None:
            start = chapters.set_index("chapter_id").loc[ch, "start_page"]
            h = h[h.page >= start]
        t = texts[(texts.atlas == p.atlas) & (texts.page == p.page) & (texts.element_id != f.caption_element)]
        near = min(t.itertuples(), key=lambda r: edge_distance((p.x0, p.y0, p.x1, p.y1), box(r)), default=None)
        out.append(dict(chapter_id=ch, section_heading=h.text.iloc[-1] if len(h) else "",
                        adjacent_paragraph=near.text if near is not None else ""))
    return pd.concat([figs.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def keep_figure(sizes, chapter_id, min_side=120, max_aspect=6):
    if not isinstance(chapter_id, str):
        return False
    return any(min(w, h) >= min_side and max(w, h) / max(min(w, h), 1) <= max_aspect for w, h in sizes)


def build_figures(pairs, elements, chapters):
    figs = group_panels(pairs)
    by_panel = pairs.set_index("panel_id")
    el = elements.set_index("element_id")
    first = by_panel.loc[[p[0] for p in figs.panel_ids]].reset_index()
    figs = figs.assign(
        atlas=first.atlas.values, page=first.page.values,
        crop_paths=[[el.crop_path[p] for p in ps] for ps in figs.panel_ids],
        caption_vlm=[el.text.get(e, "") if isinstance(e, str) else "" for e in figs.caption_element],
        caption_rule=[el.text.get(e, "") if isinstance(e, str) else "" for e in first.rule_element],
        confidence=[by_panel.loc[ps, "confidence"].mean() for ps in figs.panel_ids],
    )
    figs["figure_id"] = [f"F:{a}:{p}:{n}" for (a, p), n in zip(zip(figs.atlas, figs.page), figs.groupby(["atlas", "page"]).cumcount())]
    figs = add_context(figs, elements, chapters)
    figs["keep"] = [keep_figure([(el.x1[p] - el.x0[p], el.y1[p] - el.y0[p]) for p in ps], ch)
                    for ps, ch in zip(figs.panel_ids, figs.chapter_id)]
    cols = ["figure_id", "atlas", "page", "panel_ids", "crop_paths", "caption_element", "caption_vlm", "caption_rule",
            "confidence", "section_heading", "adjacent_paragraph", "chapter_id", "keep"]
    return figs[cols]
