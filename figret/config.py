from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Atlas:
    id: str
    file: str
    ocr: bool | str = False
    toc_level: int | None = 1


@dataclass
class Config:
    name: str
    root: Path
    pdf_dir: Path
    run_dir: Path
    dpi: int
    atlases: list[Atlas]
    models: dict = field(default_factory=dict)
    prompts: dict = field(default_factory=dict)

    def pdf(self, atlas):
        return self.pdf_dir / atlas.file


def load_config(path):
    path = Path(path).resolve()
    root = path.parent.parent
    raw = yaml.safe_load(path.read_text())
    return Config(
        name=raw["name"],
        root=root,
        pdf_dir=root / raw["pdf_dir"],
        run_dir=root / raw["run_dir"],
        dpi=raw.get("dpi", 144),
        atlases=[Atlas(**a) for a in raw["atlases"]],
        models=raw.get("models", {}),
        prompts=raw.get("prompts", {}),
    )
