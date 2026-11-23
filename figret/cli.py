import argparse
import time

from figret.config import load_config


def cmd_parse(cfg, args):
    from figret.parse import parse_atlas

    out = cfg.run_dir / "elements"
    out.mkdir(parents=True, exist_ok=True)
    pages = None
    if args.pages:
        a, b = args.pages.split("-")
        pages = range(int(a), int(b) + 1)
    for atlas in cfg.atlases:
        if args.atlas and atlas.id not in args.atlas:
            continue
        t = time.time()
        df = parse_atlas(cfg, atlas, pages)
        df.to_parquet(out / f"{atlas.id}.parquet")
        print(f"{atlas.id} pages={df.page.nunique()} {df.kind.value_counts().to_dict()} {time.time() - t:.1f}s", flush=True)


def cmd_chapters(cfg, args):
    from figret.chapters import build_chapters

    ch = build_chapters(cfg)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    ch.to_parquet(cfg.run_dir / "chapters.parquet")
    print(f"{len(ch)} chapters {ch.atlas.value_counts().to_dict()}")


def cmd_pair(cfg, args):
    import pandas as pd
    from figret import vlm
    from figret.pairing import build_figures, run_pairing
    from figret.parse import load_elements

    elements = load_elements(cfg)
    pages = sorted(set(map(tuple, elements[elements.kind == "picture"][["atlas", "page"]].values.tolist())))
    t = time.time()
    pairs = run_pairing(vlm.load(cfg.models["vlm"]), elements, pages)
    pairs.to_parquet(cfg.run_dir / "pairs.parquet")
    figs = build_figures(pairs, elements, pd.read_parquet(cfg.run_dir / "chapters.parquet"))
    figs.to_parquet(cfg.run_dir / "figures.parquet")
    print(f"{len(pages)} pages, {len(figs)} figures, {int(figs.keep.sum())} kept, "
          f"{int((figs.caption_vlm != '').sum())} captioned, {time.time() - t:.0f}s", flush=True)


def cmd_describe(cfg, args):
    import pandas as pd
    from figret import vlm
    from figret.describe import describe
    from figret.index import crop_file, kept_figures

    prompt = cfg.prompts.get("describe", "")
    if not prompt or prompt.startswith("<"):
        raise SystemExit("set prompts.describe in the config")
    out = cfg.run_dir / "descriptions.parquet"
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["figure_id", "text"])
    todo = kept_figures(cfg)
    todo = todo[~todo.figure_id.isin(done.figure_id)]
    if not len(todo):
        print(f"{len(done)} figures described")
        return
    model = vlm.load(cfg.models["vlm"])
    rows, t = [], time.time()
    for k, f in enumerate(todo.itertuples(), 1):
        rows.append(dict(figure_id=f.figure_id, text=describe(model, [crop_file(cfg, c) for c in f.crop_paths], prompt)))
        if k % 100 == 0 or k == len(todo):
            pd.concat([d for d in (done, pd.DataFrame(rows)) if len(d)], ignore_index=True).to_parquet(out)
            print(f"{k}/{len(todo)} {(time.time() - t) / k:.2f}s each", flush=True)


def cmd_embed(cfg, args):
    from figret import index
    from figret.routes import ROUTES

    has_desc = index.descriptions_ready(cfg)
    last = None
    for route in args.route or list(ROUTES):
        r = ROUTES[route]
        done = (index.index_dir(cfg) / f"{index.index_name(route)}.npy").exists()
        if r.encoder == "bm25" or ("description" in r.field and not has_desc) or (done and not args.force):
            continue
        if r.encoder != last:
            index.release()
            last = r.encoder
        t = time.time()
        index.build(cfg, route)
        print(f"{route} {time.time() - t:.1f}s", flush=True)


def cmd_search(cfg, args):
    import pandas as pd
    from figret import index
    from figret.routes import ROUTES

    caption = pd.read_parquet(cfg.run_dir / "figures.parquet").set_index("figure_id").caption_vlm
    last = None
    for route in args.route or ["caption-context"]:
        if ROUTES[route].encoder != last:
            index.release()
            last = ROUTES[route].encoder
        print(f"== {route}")
        for r in index.search(cfg, route, [args.query], k=args.k).itertuples():
            print(f"  {r.rank:2d} {r.score:.3f} {r.figure_id:24s} {caption.get(r.figure_id, '')[:90]}")


COMMANDS = {"parse": cmd_parse, "chapters": cmd_chapters, "pair": cmd_pair, "describe": cmd_describe,
            "embed": cmd_embed, "search": cmd_search}


def main():
    from figret.routes import ROUTES

    p = argparse.ArgumentParser(prog="figret")
    p.add_argument("command", choices=list(COMMANDS))
    p.add_argument("--config", required=True)
    p.add_argument("--atlas", nargs="*")
    p.add_argument("--pages")
    p.add_argument("--route", nargs="*", choices=list(ROUTES))
    p.add_argument("--force", action="store_true")
    p.add_argument("--query")
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()
    if args.command == "search" and not args.query:
        p.error("search needs --query")
    COMMANDS[args.command](load_config(args.config), args)


if __name__ == "__main__":
    main()
