from dataclasses import dataclass


@dataclass(frozen=True)
class Route:
    encoder: str
    field: str


ROUTES = {
    "caption-context": Route("text", "caption_context"),
    "caption-heading": Route("text", "caption_heading"),
    "caption": Route("text", "caption"),
    "caption-rule": Route("text", "caption_rule"),
    "bm25-caption": Route("bm25", "caption"),
    "caption+description": Route("text", "caption_description"),
    "description": Route("text", "description"),
    "pixels-clip": Route("image:clip", "image"),
    "pixels-siglip2": Route("image:siglip2", "image"),
    "pixels-medsiglip": Route("image:medsiglip", "image"),
    "pixels-biomedclip": Route("image:biomedclip", "image"),
}
