from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def write_identity_map(renders_dir: Path, out_path: Path) -> Path:
    # Map commonly generated identity images to logical roles Headshot 2 steps use
    mapping: Dict[str, str] = {
        "front": "kelly_identity_front_v001.png",
        "three_quarter": "kelly_identity_three_quarter_v001.png",
        "profile": "kelly_identity_profile_v001.png",
    }
    abs_map = {k: str((renders_dir / v).resolve().as_posix()) for k, v in mapping.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"identity_set": abs_map}, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="CC5 identity-set helper JSON writer")
    parser.add_argument("--renders", type=Path, default=Path("projects/Kelly/assets/renders"))
    parser.add_argument("--out", type=Path, default=Path("projects/Kelly/assets/cc5_identity_set.json"))
    args = parser.parse_args()
    p = write_identity_map(args.renders, args.out)
    print(str(p))


if __name__ == "__main__":
    main()


