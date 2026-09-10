"""Keep repository-local Markdown links usable after documentation moves."""

import re
from pathlib import Path


def test_local_documentation_links():
    root = Path(__file__).parents[1]
    documents = [
        *root.glob("README*.md"),
        *root.joinpath("docs").rglob("*.md"),
        root / "ckpt/README.md",
        root / "dataset/README.md",
    ]
    broken = []
    for document in documents:
        content = document.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^\s)]+)\)", content):
            if "://" in target or target.startswith("#"):
                continue
            path = target.split("#", 1)[0]
            if path and not (document.parent / path).exists():
                broken.append(f"{document.relative_to(root)}: {target}")
    assert not broken, "Broken documentation links:\n" + "\n".join(broken)
