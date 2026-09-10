"""
Step 2 — resolve merchant categories with the Anthropic API.

Step 1 (`app_text.py`) writes one JSON per slip into ./transactions_json, with a
`category` that is either set by the local model / rule table or left as "other".

This script reads those files and, for every entry still "other", sends ONLY the
payee name (`to.name`) to Claude and asks for a single category word. Nothing
else about the transaction leaves the machine — no amount, account, payer, date,
or reference.

    python resolve_categories.py            # resolve the "other" entries
    python resolve_categories.py --force    # re-resolve every file
    python resolve_categories.py --dry-run  # show what would change, write nothing

Configure with environment variables:
    ANTHROPIC_API_KEY   your API key (required)
    CATEGORY_MODEL      optional model override (default: claude-haiku-4-5-20251001)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests

JSON_DIR = Path(__file__).parent / "transactions_json"

CATEGORIES = (
    "food", "groceries", "shopping", "transport", "utilities", "health",
    "entertainment", "services", "education", "government", "transfer", "other",
)

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
MODEL = os.environ.get("CATEGORY_MODEL", "").strip() or "claude-haiku-4-5-20251001"
ENDPOINT = "https://api.anthropic.com/v1/messages"

_PROMPT = (
    "What single spending category best fits this business/merchant in Thailand?\n"
    'Merchant: "{name}"\n'
    "Answer with exactly one word from this list, nothing else: "
    + ", ".join(c for c in CATEGORIES if c != "transfer")
)

_cache: dict[str, str] = {}


def categorise(name: str) -> str | None:
    """One merchant name -> one category word (or None if the call/parse fails)."""
    if not name:
        return None
    if name in _cache:
        return _cache[name]

    try:
        resp = requests.post(
            ENDPOINT,
            headers={
                "x-api-key": API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": _PROMPT.format(name=name)}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        answer = resp.json()["content"][0]["text"]
    except (requests.RequestException, KeyError, IndexError) as e:
        print(f"    ! API call failed: {e}", file=sys.stderr)
        return None

    match = next((c for c in CATEGORIES if c in answer.strip().lower()), None)
    if match:
        _cache[name] = match
    return match


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dir", type=Path, default=JSON_DIR, help="folder of slip JSON files")
    ap.add_argument("--force", action="store_true", help="re-resolve every file, not just 'other'")
    ap.add_argument("--dry-run", action="store_true", help="print changes, write nothing")
    args = ap.parse_args()

    if not API_KEY:
        print(
            'Not configured. Set your Anthropic key:\n'
            '  $env:ANTHROPIC_API_KEY = "sk-ant-..."',
            file=sys.stderr,
        )
        return 2

    files = sorted(args.dir.glob("*.json"))
    if not files:
        print(f"No .json files in {args.dir} — run app_text.py first.", file=sys.stderr)
        return 1

    changed = resolved = skipped = 0
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        current = data.get("category", "other")
        payee = (data.get("to") or {}).get("name")

        if current != "other" and not args.force:
            skipped += 1
            print(f"{path.name}: keep {current!r}")
            continue

        new = categorise(payee)
        if not new or new == current:
            print(f"{path.name}: {payee!r} -> {current!r} (unchanged)")
            continue

        resolved += 1
        print(f"{path.name}: {payee!r}  {current!r} -> {new!r}")
        if not args.dry_run:
            data["category"] = new
            data["category_source"] = "api:anthropic"
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            changed += 1

    verb = "would change" if args.dry_run else "changed"
    print(f"\n{len(files)} files · {skipped} kept · {resolved} resolved · {changed} {verb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
