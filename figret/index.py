import json
from pathlib import Path

import numpy as np
import pandas as pd

from figret import encoders
from figret.routes import ROUTES

_cache = {}


def encoder(cfg, key):
    if key not in _cache:
        kind, _, name = key.partition(":")
        _cache[key] = encoders.text(cfg.models["text"]) if kind == "text" else encoders.image_text(cfg.models["image"][name])
    return _cache[key]


def release():
    import gc
    import torch

    _cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def index_name(route):
    r = ROUTES[route]
    return f"{r.field}-{r.encoder.replace(':', '-')}"


def index_dir(cfg):
    return cfg.run_dir / "index"


def crop_file(cfg, path):
    return str(cfg.run_dir / "crops" / Path(path).name)


def join(*parts):
    return ". ".join(p.strip().rstrip(".") for p in parts if isinstance(p, str) and p.strip())


def figure_texts(cfg, figs, field):
    desc = {}
    p = cfg.run_dir / "descriptions.parquet"
    if field in ("description", "caption_description") and p.exists():
        desc = pd.read_parquet(p).set_index("figure_id").text.to_dict()
    make = {
        "caption": lambda f: f.caption_vlm,
        "caption_rule": lambda f: f.caption_rule,
        "caption_heading": lambda f: join(f.section_heading, f.caption_vlm),
        "caption_context": lambda f: join(f.section_heading, f.caption_vlm, f.adjacent_paragraph),
        "description": lambda f: desc.get(f.figure_id, ""),
        "caption_description": lambda f: join(f.caption_vlm, desc.get(f.figure_id, "")),
    }[field]
    return [make(f) for f in figs.itertuples()]


def kept_figures(cfg):
    figs = pd.read_parquet(cfg.run_dir / "figures.parquet")
    return figs[figs.keep & figs.atlas.isin([a.id for a in cfg.atlases])]


def items(cfg, route):
    r = ROUTES[route]
    figs = kept_figures(cfg)
    if r.field == "image":
        rows = [(f.figure_id, crop_file(cfg, c)) for f in figs.itertuples() for c in f.crop_paths]
        return [x[0] for x in rows], [x[1] for x in rows]
    texts = figure_texts(cfg, figs, r.field)
    keep = [bool(t) for t in texts]
    return figs.figure_id[keep].tolist(), [t for t, k in zip(texts, keep) if k]


def descriptions_ready(cfg):
    p = cfg.run_dir / "descriptions.parquet"
    if not p.exists():
        return False
    return set(kept_figures(cfg).figure_id) <= set(pd.read_parquet(p, columns=["figure_id"]).figure_id)


def build(cfg, route):
    r = ROUTES[route]
    ids, content = items(cfg, route)
    enc = encoder(cfg, r.encoder)
    emb = enc.embed_images(content) if r.field == "image" else enc.embed(content)
    out = index_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{index_name(route)}.npy", emb.astype(np.float16))
    (out / f"{index_name(route)}.ids.json").write_text(json.dumps({"ids": ids}))


def top_k(scores, item_ids, k=10):
    inv, uniq = pd.factorize(np.asarray(item_ids))
    rows = []
    for qi in range(len(scores)):
        best = np.full(len(uniq), -np.inf)
        np.maximum.at(best, inv, scores[qi])
        order = np.argsort(-best, kind="stable")[:k]
        order = order[np.isfinite(best[order])]
        rows += [(qi, n + 1, uniq[j], float(best[j])) for n, j in enumerate(order)]
    return pd.DataFrame(rows, columns=["query_id", "rank", "figure_id", "score"])


def search(cfg, route, queries, k=10):
    r = ROUTES[route]
    queries = list(queries)
    if r.encoder == "bm25":
        return bm25_search(cfg, route, queries, k)
    ids = json.loads((index_dir(cfg) / f"{index_name(route)}.ids.json").read_text())["ids"]
    emb = np.load(index_dir(cfg) / f"{index_name(route)}.npy").astype(np.float32)
    enc = encoder(cfg, r.encoder)
    vec = enc.embed(queries, query=True) if r.encoder == "text" else enc.embed_texts(queries)
    return top_k(vec @ emb.T, ids, k)


def bm25_search(cfg, route, queries, k):
    import bm25s

    ids, texts = items(cfg, route)
    bm = bm25s.BM25()
    bm.index(bm25s.tokenize(texts, stopwords="en", show_progress=False), show_progress=False)
    q_tokens = tokens_as_lists(bm25s.tokenize(queries, stopwords="en", show_progress=False))
    scores = np.full((len(queries), len(ids)), -np.inf, dtype=np.float32)
    for i, tokens in enumerate(q_tokens):
        if tokens:
            s = bm.get_scores(tokens)
            scores[i] = np.where(s > 0, s, -np.inf)
    return top_k(scores, ids, k)


def tokens_as_lists(tokenized):
    vocab = {v: k for k, v in tokenized.vocab.items()}
    return [[vocab[i] for i in ids] for ids in tokenized.ids]
