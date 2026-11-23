from PIL import Image

MULTI = "The images are panels of one figure. "


def describe(vlm, paths, prompt, max_panels=4):
    ims = [Image.open(p) for p in paths[:max_panels]]
    return vlm.ask(ims, (MULTI if len(ims) > 1 else "") + prompt, max_new_tokens=200)
