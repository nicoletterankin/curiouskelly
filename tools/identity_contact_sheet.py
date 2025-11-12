from __future__ import annotations

from pathlib import Path
from typing import List
from PIL import Image, ImageDraw, ImageFont


def make_contact_sheet(images: List[Path], out_path: Path, cols: int = 3, cell_px: int = 768) -> None:
    imgs = []
    labels = []
    for p in images:
        if not p.exists():
            continue
        img = Image.open(p).convert("RGB")
        img = img.copy()
        img.thumbnail((cell_px, cell_px), Image.LANCZOS)
        imgs.append(img)
        labels.append(p.name)

    if not imgs:
        raise RuntimeError("No images provided for contact sheet")

    rows = (len(imgs) + cols - 1) // cols
    margin = 16
    label_h = 28
    w = cols * cell_px + (cols + 1) * margin
    h = rows * (cell_px + label_h) + (rows + 1) * margin
    sheet = Image.new("RGB", (w, h), color=(24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    font = None
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    x = margin
    y = margin
    for idx, (im, label) in enumerate(zip(imgs, labels)):
        sheet.paste(im, (x, y))
        # label
        text_pos = (x, y + cell_px + 4)
        draw.text(text_pos, label, fill=(220, 220, 220), font=font)
        x += cell_px + margin
        if (idx + 1) % cols == 0:
            x = margin
            y += cell_px + label_h + margin

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Identity contact sheet")
    parser.add_argument("--renders", type=Path, default=Path("projects/Kelly/assets/renders"))
    parser.add_argument("--out", type=Path, default=Path("projects/Kelly/assets/identity_contact_sheet.png"))
    args = parser.parse_args()

    # Discover identity images by pattern and role
    def find_one(role: str):
        # Prefer studio_neutral variants, else any
        prefer = list(args.renders.glob(f"kelly_identity_{role}_*_v*.png"))
        if not prefer:
            prefer = list(args.renders.glob(f"kelly_identity_{role}_v*.png"))
        return prefer[0] if prefer else None

    candidates = [find_one("front"), find_one("three_quarter"), find_one("profile")]
    paths = [p for p in candidates if p]
    make_contact_sheet(paths, args.out)
    print(str(args.out))


if __name__ == "__main__":
    main()


