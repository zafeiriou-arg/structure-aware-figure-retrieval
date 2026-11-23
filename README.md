# figret

Ingest PDF books into a figure index and search it with text.

```
pip install -e .
```

## Configure

Copy `configs/example.yaml` and fill in the placeholders:

- `atlases`: one entry per PDF in `pdf_dir`. `toc_level` is the outline level that marks chapters (`null` for none);
  `ocr: vlm` reads a scanned book with the vision-language model.
- `models.layout`: weights of a YOLO layout detector trained on DocLayNet (Ultralytics format).
- `prompts.describe`: the instruction the vision-language model gets for one figure image. Needed only for the
  `description` and `caption+description` routes.

Chapters come from the PDF outline, or from `<pdf_dir>/chapters_manual.csv` (columns `atlas, chapter_id, title,
start_page, end_page`, pages counted from 0). Figures outside a chapter are not indexed.

## Run

```
figret parse    --config configs/my.yaml [--pages 0-49] [--atlas book1]
figret chapters --config configs/my.yaml
figret pair     --config configs/my.yaml
figret describe --config configs/my.yaml
figret embed    --config configs/my.yaml [--route caption-context ...] [--force]
figret search   --config configs/my.yaml --query "..." [--route caption-context ...] [--k 10]
```

`pair` and `describe` need a GPU with room for the vision-language model (about 17 GB).

## Routes

| Route | Indexed |
|---|---|
| `caption-context` | section heading, caption and nearest paragraph |
| `caption-heading` | section heading and caption |
| `caption` | caption |
| `caption-rule` | caption chosen by the nearest-caption rule |
| `bm25-caption` | caption, BM25 |
| `caption+description` | caption and generated description |
| `description` | generated description |
| `pixels-clip`, `pixels-siglip2`, `pixels-medsiglip`, `pixels-biomedclip` | figure image |
