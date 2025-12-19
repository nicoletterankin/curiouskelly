import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LESSONS_DIR = REPO_ROOT / "public" / "lessons"

REQUIRED_PHASES = ["hook", "cliff", "q1", "q2", "q3", "wisdom", "outro"]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_one(day: int):
    path = LESSONS_DIR / f"day-{day}.json"
    if not path.exists():
        return [f"day-{day}: missing file {path.as_posix()}"]

    try:
        data = load_json(path)
    except Exception as e:
        return [f"day-{day}: invalid json ({e})"]

    issues: list[str] = []
    phases = data.get("phases") or {}

    for key in REQUIRED_PHASES:
        node = phases.get(key)
        if not isinstance(node, dict):
            issues.append(f"day-{day}: missing phase '{key}'")
            continue

        opts = node.get("options")
        if not isinstance(opts, list) or len(opts) != 2:
            issues.append(f"day-{day}: phase '{key}' options expected 2, got {0 if not isinstance(opts, list) else len(opts)}")

        # Require text+response in EN at minimum (translations can be placeholders).
        for idx, opt in enumerate(opts or []):
            if not isinstance(opt, dict):
                issues.append(f"day-{day}: phase '{key}' option[{idx}] not an object")
                continue
            text = (opt.get("text") or {}).get("en") if isinstance(opt.get("text"), dict) else opt.get("text")
            resp = (opt.get("response") or {}).get("en") if isinstance(opt.get("response"), dict) else opt.get("response")
            if not text:
                issues.append(f"day-{day}: phase '{key}' option[{idx}] missing text.en")
            if not resp:
                issues.append(f"day-{day}: phase '{key}' option[{idx}] missing response.en")

        script = node.get("script")
        script_en = script.get("en") if isinstance(script, dict) else script
        if not script_en:
            issues.append(f"day-{day}: phase '{key}' missing script.en")

    meta = data.get("meta") or {}
    target = meta.get("target_audience")
    if target and target != "adult":
        issues.append(f"day-{day}: target_audience expected 'adult', got '{target}'")

    return issues


def main() -> int:
    if not LESSONS_DIR.exists():
        print(f"ERROR: missing lessons dir: {LESSONS_DIR.as_posix()}")
        return 2

    all_issues: list[str] = []
    for day in range(1, 366):
        all_issues.extend(validate_one(day))

    if all_issues:
        print("Seed lesson validation FAILED:\n")
        for line in all_issues[:500]:
            print(f"- {line}")
        if len(all_issues) > 500:
            print(f"\n... plus {len(all_issues) - 500} more")
        return 1

    print("Seed lesson validation PASSED (365 days, 7 phases, 2 choices each).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
