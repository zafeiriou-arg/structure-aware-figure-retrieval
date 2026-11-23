import fitz
import pandas as pd

COLS = ["atlas", "chapter_id", "title", "start_page", "end_page"]


def chapters_from_toc(atlas_id, toc, level, n_pages):
    entries = [(title.strip(), page - 1) for lvl, title, page, *_ in toc if lvl == level and page > 0]
    rows = []
    for k, (title, start) in enumerate(entries):
        end = entries[k + 1][1] - 1 if k + 1 < len(entries) else n_pages - 1
        rows.append((atlas_id, f"{atlas_id}:{k + 1:02d}", title, start, max(start, end)))
    return pd.DataFrame(rows, columns=COLS)


def build_chapters(cfg):
    manual_path = cfg.pdf_dir / "chapters_manual.csv"
    manual = pd.read_csv(manual_path) if manual_path.exists() else pd.DataFrame(columns=COLS)
    parts = []
    for a in cfg.atlases:
        if a.id in set(manual.atlas):
            parts.append(manual[manual.atlas == a.id])
        elif a.toc_level:
            doc = fitz.open(cfg.pdf(a))
            parts.append(chapters_from_toc(a.id, doc.get_toc(), a.toc_level, len(doc)))
    return pd.concat(parts, ignore_index=True)[COLS] if parts else pd.DataFrame(columns=COLS)


def chapter_of(chapters, atlas, page):
    hit = chapters[(chapters.atlas == atlas) & (chapters.start_page <= page) & (chapters.end_page >= page)]
    return hit.chapter_id.iloc[0] if len(hit) else None
