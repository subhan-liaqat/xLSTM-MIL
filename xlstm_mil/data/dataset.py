from __future__ import annotations

import math
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _norm_id(s):
    s = str(s).strip().lower()
    s = Path(s).stem
    return re.sub(r"[^a-z0-9]+", "", s)


def infer_label_from_slide_id(slide_id_stem):
    s = slide_id_stem.lower()
    if "tumor" in s:
        return 1
    if "normal" in s:
        return 0
    return None


def infer_nsclc_label(slide_id_stem, row=None):
    sid = str(slide_id_stem).lower()
    row_vals = []
    if row is not None:
        for key in ["project_id", "project", "primary_diagnosis", "diagnosis", "label", "class"]:
            if key in row and pd.notna(row[key]):
                row_vals.append(str(row[key]).lower())
    haystack = " ".join([sid] + row_vals)

    if "luad" in haystack or "adenocarcinoma" in haystack:
        return 1
    if "lusc" in haystack or "squamous" in haystack:
        return 0
    return None


def infer_tcga_patient_id(slide_id_stem):
    sid = str(slide_id_stem).strip()
    m = re.search(r"(TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4})", sid, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    parts = sid.split("-")
    if len(parts) >= 3 and parts[0].lower() == "tcga":
        return "-".join(parts[:3]).upper()
    return sid.lower()


def _xy2d(n_side, x, y):
    d = 0
    s = n_side // 2
    while s > 0:
        rx = 1 if (x & s) else 0
        ry = 1 if (y & s) else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = n_side - 1 - x
                y = n_side - 1 - y
            x, y = y, x
        s //= 2
    return d


def hilbert_indices(coords_np):
    try:
        import hilbertsfc

        mx = int(coords_np.max()) if coords_np.size else 0
        p = max(1, int(math.ceil(math.log2(mx + 1))))
        if hasattr(hilbertsfc, "hilbert_index"):
            return np.array(
                [hilbertsfc.hilbert_index(int(x), int(y), p=p) for x, y in coords_np],
                dtype=np.int64,
            )
        if hasattr(hilbertsfc, "encode"):
            try:
                return np.array(hilbertsfc.encode(coords_np.astype(np.int64)), dtype=np.int64)
            except Exception:
                return np.array(hilbertsfc.encode(coords_np.astype(np.int64).T), dtype=np.int64)
    except Exception:
        pass
    mx = int(coords_np.max()) if coords_np.size else 0
    p = max(1, int(math.ceil(math.log2(mx + 1))))
    n_side = 1 << p
    return np.array([_xy2d(n_side, int(x), int(y)) for x, y in coords_np.astype(np.int64)], dtype=np.int64)


class WSIFeatureDataset(Dataset):
    def __init__(self, embed_dir, patch_dir, process_csv, task: str = "camelyon16"):
        self.embed_dir = Path(embed_dir)
        self.patch_dir = Path(patch_dir)
        self.task = str(task).strip().lower()
        if self.task not in {"camelyon16", "tcga_nsclc"}:
            raise ValueError(f"Unsupported task '{task}'. Expected one of: camelyon16, tcga_nsclc")

        df = pd.read_csv(process_csv)
        if "slide_id" not in df.columns:
            raise ValueError(f"'slide_id' missing. Columns: {list(df.columns)}")

        split_col = next((c for c in ["split", "set", "subset", "partition"] if c in df.columns), None)

        pt_files = sorted(self.embed_dir.glob("*.pt"))
        h5_files = sorted(self.patch_dir.rglob("*.h5"))
        pt_map = {_norm_id(p.stem): p for p in pt_files}
        h5_map = {_norm_id(p.stem): p for p in h5_files}

        self.samples = []
        for _, row in df.iterrows():
            sid_stem = Path(str(row["slide_id"]).strip()).stem
            if self.task == "camelyon16":
                label = infer_label_from_slide_id(sid_stem)
            else:
                label = infer_nsclc_label(sid_stem, row)
            if label is None:
                continue

            if split_col is not None and pd.notna(row[split_col]):
                split_name = str(row[split_col]).strip().lower()
            else:
                if self.task == "camelyon16":
                    sid_lower = sid_stem.lower()
                    if sid_lower.startswith("test_"):
                        split_name = "test"
                    elif sid_lower.startswith(("normal_", "tumor_")):
                        split_name = "train"
                    else:
                        split_name = "unknown"
                else:
                    split_name = "unspecified"

            key = _norm_id(sid_stem)
            pt_path = pt_map.get(key)
            h5_path = h5_map.get(key)
            if pt_path is None or h5_path is None:
                continue
            patient_id = infer_tcga_patient_id(sid_stem) if self.task == "tcga_nsclc" else sid_stem
            self.samples.append((sid_stem, pt_path, h5_path, int(label), split_name, patient_id))

        if len(self.samples) == 0:
            raise RuntimeError("No matched samples found after pairing manifest <-> pt <-> h5")

        self._order_cache = {}

        y = np.array([s[3] for s in self.samples], dtype=np.int64)
        split_counts = pd.Series([s[4] for s in self.samples]).value_counts().to_dict()
        if self.task == "camelyon16":
            print(f"Loaded {len(self.samples)} slides | normal={(y==0).sum()} tumor={(y==1).sum()}")
        else:
            unique_patients = len({s[5] for s in self.samples})
            print(f"Loaded {len(self.samples)} slides | LUSC={(y==0).sum()} LUAD={(y==1).sum()} | patients={unique_patients}")
        print(f"Split counts: {split_counts}")

    def __len__(self):
        return len(self.samples)

    def indices_for_split(self, split_name):
        target = str(split_name).strip().lower()
        return np.array([i for i, s in enumerate(self.samples) if s[4] == target], dtype=np.int64)

    def patient_ids(self):
        return [s[5] for s in self.samples]

    def _read_coords_h5(self, h5_path, sid=""):
        with h5py.File(h5_path, "r") as f:
            if "coords" in f:
                coords = f["coords"][:]
            else:
                coords = None
                for k in f.keys():
                    arr = f[k][:]
                    if arr.ndim == 2 and arr.shape[1] == 2:
                        coords = arr
                        break
                if coords is None:
                    raise KeyError(f"{sid}: no coordinate dataset (N,2) found")
        return np.asarray(coords, dtype=np.float64)

    def _hilbert_order_for_slide(self, sid, h5_path, n_feats):
        if sid in self._order_cache:
            expected_n, order = self._order_cache[sid]
            if n_feats != expected_n:
                raise ValueError(
                    f"{sid}: features length changed ({n_feats} != cached {expected_n}); possible data mismatch"
                )
            return order

        coords = self._read_coords_h5(h5_path, sid)
        if coords.shape[0] != n_feats:
            raise ValueError(
                f"{sid}: mismatched patch counts feat={n_feats} coord={coords.shape[0]}; refusing to silently truncate"
            )
        if n_feats == 0:
            raise RuntimeError(f"{sid}: empty bag")

        order = np.argsort(hilbert_indices(coords.astype(np.int64, copy=False)))
        self._order_cache[sid] = (n_feats, order)
        return order

    def load_bag_with_coords(self, idx):
        sid, pt_path, h5_path, label, split_name, _patient_id = self.samples[idx]
        try:
            feats = torch.load(pt_path, map_location="cpu", weights_only=False)
        except TypeError:
            feats = torch.load(pt_path, map_location="cpu")
        if isinstance(feats, dict):
            feats = feats["features"] if "features" in feats else next(iter(feats.values()))
        feats = torch.as_tensor(feats, dtype=torch.float32)
        if feats.ndim != 2:
            raise ValueError(f"{sid}: expected 2D features, got {tuple(feats.shape)}")

        order = self._hilbert_order_for_slide(sid, h5_path, feats.shape[0])
        ot = torch.from_numpy(order)
        feats = feats[ot]

        coords_np = self._read_coords_h5(h5_path, sid)
        if coords_np.shape[0] != feats.shape[0]:
            raise ValueError(
                f"{sid}: coord count {coords_np.shape[0]} != feat count {feats.shape[0]} after ordering"
            )
        coords = torch.from_numpy(coords_np)[ot]

        y = torch.tensor([float(label)], dtype=torch.float32)
        return {"slide_id": sid, "features": feats, "coords": coords, "label": y, "split": split_name}

    def __getitem__(self, idx):
        bag = self.load_bag_with_coords(idx)
        return {"slide_id": bag["slide_id"], "features": bag["features"], "label": bag["label"], "split": bag["split"]}


def infer_feature_dim(dataset: WSIFeatureDataset) -> int:
    return int(dataset[0]["features"].shape[1])
