from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


def download_camelyon16_uni_layout(
    data_root: Path | str,
    repo_id: str = "kaczmarj/camelyon16-uni",
    token: str | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Download HF snapshot. Returns repo_root, embed_dir, patch_dir, process_csv."""
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    if token is None:
        token = os.environ.get("HF_TOKEN")
    local_repo = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            local_dir=str(root),
        )
    )
    embed_dir = local_repo / "embeddings"
    patch_dir = local_repo / "patches"
    process_csv = patch_dir / "process_list_autogen.csv"
    print("Repo root:", local_repo)
    print("Embeddings dir:", embed_dir)
    print("Patches dir:", patch_dir)
    print("Manifest:", process_csv)
    print("#pt:", len(list(embed_dir.glob("*.pt"))))
    print("#h5:", len(list(patch_dir.rglob("*.h5"))))
    return local_repo, embed_dir, patch_dir, process_csv
