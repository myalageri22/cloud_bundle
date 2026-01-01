# CNN2.py
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence, Set
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from functools import partial

# All imports
import io
import logging
import sys
import os
import json
import zipfile
import re
import random
import shutil
import argparse
import time
import subprocess

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "PyTorch is required to run this training pipeline. "
        "Install it with `pip install torch` (pick the wheel matching your Python version and platform)."
    ) from exc
import torch.nn as nn


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(deterministic)
    except (AttributeError, RuntimeError):
        if deterministic:
            logging.getLogger(__name__).warning(
                "torch.use_deterministic_algorithms unavailable; continuing without strict determinism."
            )


def _seed_worker(worker_id: int, base_seed: int) -> None:
    "Top-level worker seeding function (required for pickleability with spawn)."
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_worker_init_fn(base_seed: int):
    "Return a top-level, picklable worker init fn for DataLoader (spawn-safe on macOS)."
    return partial(_seed_worker, base_seed=base_seed)

try:
    import nibabel as nib
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The `nibabel` package is needed to read NIfTI files. Install it with `pip install nibabel`."
    ) from exc
class Config:
    "Centralized configuration for training and model parameters."
    def __init__(self):
        # basic runtime
        self.project_root = Path(__file__).resolve().parent
        self.data_dir = self.project_root / "data"
        # Default dataset root where <id>.img.nii[.gz] and <id>.label.nii.gz live
        self.data_root = self.project_root
        self.checkpoint_dir = self.project_root / "checkpoints2"
        self.log_dir = self.project_root / "logs"
        self.processed_dir = self.data_dir / "processed"
        self.device = self._select_device()

        # data / transform
        self.pixdim = (1.0, 1.0, 1.0)
        self.modality = "mri"
        self.dataset_preset = "custom"
        self.dataset = "custom"
        # Larger patch for more vascular context (raise/decrease if OOM)
        self.roi_size = (96, 192, 192)

        # training defaults
        self.batch_size = 2
        self.learning_rate = 2e-4
        self.weight_decay = 1e-5
        cpu_count = os.cpu_count() or 1
        # Windows multiprocessing DataLoader workers frequently crash; default to 0 there.
        self.num_workers = 0 if os.name == "nt" else min(4, cpu_count)
        self.pin_memory = self.device.startswith("cuda")
        self.accumulation_steps = 1
        self.use_attention = False
        self.grad_clip_norm = 1.0

        # model hyperparameters (customizable via CLI)
        # Default UNet width (matches existing checkpoints)
        self.unet_channels = (32, 64, 128, 256, 512)
        self.unet_strides = (2, 2, 2, 2)
        self.unet_num_res_units = 3
        self.unet_dropout = 0.15
        self.unet_norm = "instance"

        # data loading/runtime controls
        self.prefetch_factor = 2
        self.persistent_workers = True
        self.compile_model = False
        self.use_tta = False
        # Binarization threshold for converting probabilities to binary predictions.
        # Default 0.5, but may need lowering if model predicts low confidences.
        self.prob_threshold = 0.5
        # CT window (Hounsfield Units) used when modality == "ct".
        # Override via dataset preset or future CLI if needed.
        self.ct_window = (-100, 700)
        # caching controls
        self.cache_rate_train = 0.25  # fraction of train set to cache in memory
        self.cache_rate_val = 1.0     # validation is small; cache fully by default

        # scheduler
        self.scheduler_t0 = 8
        self.scheduler_tmult = 2
        self.scheduler_eta_min = 1e-6

        # files
        self.split_file = self.project_root / "splits.json"
        # ImageCAS options
        self.imagecas_root: Optional[Path] = None
        self.imagecas_version: str = "v2-latest"
        self.split_xlsx: Optional[Path] = None
        self.use_official_split: bool = False
        self.split_id: int = 1

        # ensure dirs
        for d in (self.data_dir, self.checkpoint_dir, self.log_dir, self.processed_dir):
            d.mkdir(parents=True, exist_ok=True)

    def _select_device(self) -> str:
        "Prefer Apple Metal (MPS) on Macs, fall back to CUDA or CPU elsewhere."
        if torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def save(self, path: Path):
        d = {k: (list(v) if isinstance(v, tuple) else str(v) if isinstance(v, Path) else v)
             for k, v in self.__dict__.items() if not k.startswith("_")}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
    @classmethod
    def from_args(cls, args: argparse.Namespace):
        cfg = cls()

        # 1) Apply dataset preset defaults first (explicit CLI overrides come later)
        preset = getattr(args, "dataset_preset", None)
        if preset:
            cfg.dataset_preset = str(preset).lower()

        # Dataset preset: ImageCAS (coronary artery segmentation on CTA / CT)
        if cfg.dataset_preset == "imagecas":
            cfg.dataset = "imagecas"
            cfg.modality = "ct"
            cfg.pixdim = (0.6, 0.6, 0.6)
            cfg.roi_size = (96, 160, 160)
            cfg.learning_rate = 2e-4
            cfg.batch_size = 1  # safer for large 3D CTA volumes
            cfg.cache_rate_train = 0.10
            cfg.cache_rate_val = 0.25
            cfg.prob_threshold = 0.30
            cfg.ct_window = (-200, 700)

        if args.batch_size:
            cfg.batch_size = args.batch_size
        if args.learning_rate is not None:
            cfg.learning_rate = args.learning_rate
        if getattr(args, "weight_decay", None) is not None:
            cfg.weight_decay = args.weight_decay
        if args.roi_size:
            cfg.roi_size = tuple(int(x) for x in args.roi_size.split(","))
        if args.num_workers is not None:
            cfg.num_workers = args.num_workers
        if getattr(args, "grad_clip_norm", None) is not None:
            cfg.grad_clip_norm = args.grad_clip_norm
        if getattr(args, "prefetch_factor", None) is not None:
            cfg.prefetch_factor = max(1, args.prefetch_factor)
        if getattr(args, "persistent_workers", None) is not None:
            cfg.persistent_workers = bool(args.persistent_workers)
        if getattr(args, "compile_model", False):
            cfg.compile_model = True
        if getattr(args, "dataset", None):
            cfg.dataset = args.dataset.lower()
        if getattr(args, "unet_channels", None):
            cfg.unet_channels = tuple(int(x.strip()) for x in args.unet_channels.split(",") if x.strip())
        if getattr(args, "unet_strides", None):
            cfg.unet_strides = tuple(int(x.strip()) for x in args.unet_strides.split(",") if x.strip())
        if len(cfg.unet_channels) != len(cfg.unet_strides) + 1:
            raise ValueError(
                "unet_channels` must have exactly one more entry than unet_strides "
                f"(got {len(cfg.unet_channels)} channels vs {len(cfg.unet_strides)} strides)."
            )
        if getattr(args, "unet_res_units", None) is not None:
            cfg.unet_num_res_units = args.unet_res_units
        if getattr(args, "unet_dropout", None) is not None:
            cfg.unet_dropout = args.unet_dropout
        if getattr(args, "scheduler_t0", None) is not None:
            cfg.scheduler_t0 = max(1, args.scheduler_t0)
        if getattr(args, "scheduler_tmult", None) is not None:
            cfg.scheduler_tmult = args.scheduler_tmult
        if getattr(args, "scheduler_eta_min", None) is not None:
            cfg.scheduler_eta_min = args.scheduler_eta_min
        if getattr(args, "cache_rate_train", None) is not None:
            cfg.cache_rate_train = max(0.0, min(1.0, args.cache_rate_train))
        if getattr(args, "cache_rate_val", None) is not None:
            cfg.cache_rate_val = max(0.0, min(1.0, args.cache_rate_val))
        if getattr(args, "use_tta", False):
            cfg.use_tta = True
        if getattr(args, "prob_threshold", None) is not None:
            cfg.prob_threshold = args.prob_threshold
        if getattr(args, "modality", None):
            cfg.modality = args.modality.lower()
        if getattr(args, "dataset_preset", None):
            cfg.dataset_preset = args.dataset_preset
        if getattr(args, "imagecas_root", None):
            cfg.imagecas_root = Path(args.imagecas_root)
        if getattr(args, "data_root", None):
            cfg.data_root = Path(args.data_root)
        if getattr(args, "imagecas_version", None):
            cfg.imagecas_version = args.imagecas_version
        if getattr(args, "split_xlsx", None):
            cfg.split_xlsx = Path(args.split_xlsx)
        if getattr(args, "use_official_split", False):
            cfg.use_official_split = True
        if getattr(args, "split_id", None):
            cfg.split_id = int(args.split_id)

        return cfg


class CropByCenterd:
    """Custom transform: crop around precomputed label center to ensure patches include foreground."""
    def __init__(self, keys, label_key, roi_size):
        self.keys = keys
        self.label_key = label_key
        self.roi_size = roi_size
        bbox_file = Path(__file__).parent / "label_bboxes.json"
        if bbox_file.exists():
            with open(bbox_file) as f:
                self.bboxes = json.load(f)
        else:
            self.bboxes = {}
    
    def __call__(self, data):
        # Try multiple ways to extract filename
        filename = None
        try:
            # Try filename_or_obj (list)
            filename = data.get("label_meta_dict", {}).get("filename_or_obj", [None])[0]
            if filename:
                filename = Path(filename).name
        except Exception:
            pass
        
        if not filename:
            try:
                # Try filename field directly
                filename = data.get("label_meta_dict", {}).get("filename", None)
                if filename:
                    filename = Path(filename).name
            except Exception:
                pass
        
        # Apply crop if we have a bbox
        if filename and filename in self.bboxes:
            bbox_info = self.bboxes[filename]
            center = bbox_info.get("center")
            if center:
                for key in self.keys:
                    img = data[key]
                    img_arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
                    if img_arr.ndim == 3:
                        img_arr = img_arr[np.newaxis, ...]
                    
                    C, D, H, W = img_arr.shape
                    cz, cy, cx = center
                    rz, ry, rx = self.roi_size[0] // 2, self.roi_size[1] // 2, self.roi_size[2] // 2
                    
                    z_start = max(0, int(cz - rz))
                    z_end = min(D, int(cz + rz))
                    y_start = max(0, int(cy - ry))
                    y_end = min(H, int(cy + ry))
                    x_start = max(0, int(cx - rx))
                    x_end = min(W, int(cx + rx))
                    
                    cropped = img_arr[:, z_start:z_end, y_start:y_end, x_start:x_end]
                    pads = [(0, 0), (max(0, rz - int(cz - z_start)), max(0, rz - int(z_end - cz))), (max(0, ry - int(cy - y_start)), max(0, ry - int(y_end - cy))), (max(0, rx - int(cx - x_start)), max(0, rx - int(x_end - cx)))]
                    cropped = np.pad(cropped, pads, mode="constant", constant_values=0)
                    cropped = cropped[:, :self.roi_size[0], :self.roi_size[1], :self.roi_size[2]]
                    data[key] = torch.from_numpy(cropped).float()
        return data


def setup_logging(log_dir: Path, experiment_name: str = None) -> logging.Logger:
    "Configure logging to file and console"
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = log_dir / f"train_{experiment_name}.log"

    # Create logger
    logger = logging.getLogger("VascularSeg")
    # Default level is INFO; can be overridden by environment variable DEBUG_METRICS=1
    logger.setLevel(logging.INFO)

    # avoid duplicate handlers if called multiple times
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        # Allow runtime override to DEBUG for troubleshooting: set DEBUG_METRICS=1
        if os.getenv("DEBUG_METRICS") == "1":
            console_handler.setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)
        else:
            console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_format)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)

        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


# Data

class LocalDataManager:
    "Handles local dataset extraction and organization"
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def extract_dataset(self, zip_path: Path) -> Path:
        zip_path = Path(zip_path)
        # Directory containing multiple zip parts (ImageCAS)
        if zip_path.is_dir():
            zip_files = list(zip_path.glob("*.zip"))
            if zip_files:
                unified = self.config.processed_dir / "imagecas_extracted"
                if unified.exists() and any(unified.rglob("*.nii*")):
                    self.logger.info("Using existing extracted ImageCAS directory at %s", unified)
                    return unified
                unified.mkdir(parents=True, exist_ok=True)
                self.logger.info("Found %d zip files in %s. Extracting to %s", len(zip_files), zip_path, unified)
                for zp in zip_files:
                    self.logger.info("Extracting %s ...", zp.name)
                    with zipfile.ZipFile(zp, "r") as zf:
                        zf.extractall(unified)
                # optional flatten if needed (pairing uses rglob so structure can remain)
                return unified
        if zip_path.is_file() and zipfile.is_zipfile(zip_path):
            dest = self.config.processed_dir / zip_path.stem
            if dest.exists():
                self.logger.info(f"Using existing extracted dataset at {dest}")
                return dest
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(dest)
            # flatten any nested folders
            self._flatten_directory(dest)
            return dest
        elif zip_path.is_dir():
            return zip_path
        else:
            raise FileNotFoundError(f"Dataset not found: {zip_path}")

    def _flatten_directory(self, path: Path):
        # move any .nii/.nii.gz up to top-level processed dir for simplicity
        for p in path.rglob("*"):
            if p.is_file() and p.suffix in [".nii", ".gz"] and ("nii" in p.name):
                target = path / p.name
                if p.resolve() != target.resolve():
                    shutil.move(str(p), str(target))

    def find_nifti_files(self, root_dir: Path) -> List[Path]:
        root = Path(root_dir)
        files = [p for p in root.rglob("*") if p.is_file() and (p.suffix in [".nii", ".gz"] and "nii" in p.name)]
        return files


class DatasetPairer:
    """Intelligently pairs images with labels using regex"""
    def __init__(self, logger: logging.Logger, dataset: str = "custom"):
        self.logger = logger
        self.dataset = dataset.lower()
        # common patterns to extract id
        self.image_patterns = [
            r"(?P<id>\d+)_image",
            r"(?P<id>sub-\d+)",
            r"(?P<id>patient\d+)",
            r"(?P<id>pat\d+)_orig",
        ]
        self.label_patterns = [
            r"(?P<id>\d+)_label",
            r"(?P<id>sub-\d+)",
            r"(?P<id>patient\d+)",
            r"(?P<id>pat\d+)_orig_seg",
        ]

    def extract_id(self, filepath: Path, patterns: List[str]) -> Optional[str]:
        name = filepath.stem
        for pat in patterns:
            m = re.search(pat, name)
            if m:
                return m.group("id")
        # fallback: filename without suffix
        return name

    def _pair_imagecas(self, all_files: List[Path]) -> List[Dict[str, str]]:
        label_tokens = ("label", "mask", "seg", "gt", "coronary", "artery")
        image_tokens = ("image", "cta", "ct", "volume", "img")

        def _is_label(name: str) -> bool:
            n = name.lower()
            return any(tok in n for tok in label_tokens)

        def _is_image(name: str) -> bool:
            n = name.lower()
            return any(tok in n for tok in image_tokens) and not _is_label(name)

        def _case_id(p: Path) -> str:
            stem = re.sub(r"\.nii(\.gz)?$", "", p.name, flags=re.IGNORECASE)
            stem = re.sub(r"(image|img|cta|ct|volume|mask|label|seg|gt)$", "", stem, flags=re.IGNORECASE)
            return re.sub(r"[^0-9A-Za-z]+$", "", stem)

        img_map = defaultdict(list)
        lbl_map = defaultdict(list)
        for p in all_files:
            if _is_label(p.name):
                lbl_map[_case_id(p)].append(p)
            elif _is_image(p.name):
                img_map[_case_id(p)].append(p)
            else:
                # fallback heuristics: put as image if nothing else
                img_map[_case_id(p)].append(p)

        paired = []
        for cid in sorted(set(img_map.keys()) & set(lbl_map.keys())):
            imgs = sorted(img_map[cid], key=lambda x: (not x.name.endswith(".nii.gz"), x.name))
            lbls = sorted(lbl_map[cid], key=lambda x: (not x.name.endswith(".nii.gz"), x.name))
            paired.append({"id": cid, "image": str(imgs[0]), "label": str(lbls[0])})
        self.logger.info("Found %d ImageCAS pairs", len(paired))
        return paired

    def pair_files(self, all_files: List[Path]) -> List[Dict[str, str]]:
        if self.dataset == "imagecas":
            return self._pair_imagecas(all_files)
        images = []
        labels = []
        for p in all_files:
            n = p.name.lower()
            # ignore endpoint masks entirely so they are neither inputs nor labels
            if "seg_endpoints" in n:
                continue
            if ("label" in n or "mask" in n or "seg" in n):
                labels.append(p)
            else:
                images.append(p)
        # map ids
        id_map = defaultdict(dict)
        for img in images:
            _id = self.extract_id(img, self.image_patterns)
            id_map[_id]["image"] = str(img)
        for lab in labels:
            _id = self.extract_id(lab, self.label_patterns)
            id_map[_id]["label"] = str(lab)
        paired = []
        for k, v in id_map.items():
            if "image" in v and "label" in v:
                paired.append({"image": v["image"], "label": v["label"], "id": k})
            else:
                self.logger.debug(f"Skipping incomplete pair for id {k}: {v.keys()}")
        self.logger.info(f"Found {len(paired)} paired items")
        return paired


class DatasetSplitter:
    """Handles train/validation/test splitting"""
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def split_dataset(
        self,
        paired_data: List[Dict],
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        random_seed: int = 42
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        rng = random.Random(random_seed)
        items = paired_data.copy()
        rng.shuffle(items)
        n = len(items)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        test = items[:n_test] if n_test > 0 else []
        val = items[n_test:n_test + n_val] if n_val > 0 else items[:n_val]
        train = items[n_test + n_val:]
        # persist splits
        self._save_split(train, val, test)
        self.logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test

    def _save_split(self, train, val, test):
        out = {"train": train, "val": val, "test": test}
        with open(self.config.split_file, "w") as f:
            json.dump(out, f, indent=2)

    def _load_split(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        if not self.config.split_file.exists():
            return [], [], []
        with open(self.config.split_file) as f:
            d = json.load(f)
        return d.get("train", []), d.get("val", []), d.get("test", [])


def _parse_numeric_id(value) -> Optional[int]:
    "Best-effort numeric id extraction from ids, dicts, or filenames."
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except Exception:
            return None
    if isinstance(value, dict):
        for key in ("id", "image", "label"):
            if key in value:
                v = _parse_numeric_id(value[key])
                if v is not None:
                    return v
        return None
    name = str(value)
    name = Path(name).name
    if ".img" in name:
        name = name.split(".img", 1)[0]
    elif ".label" in name:
        name = name.split(".label", 1)[0]
    # grab first numeric token
    m = re.search(r"(\d+)", name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def discover_cloud_samples(data_root: Path, logger: logging.Logger) -> List[Dict[str, object]]:
    """
    Recursively scan data_root for <id>.img.nii[.gz] files and pair them
    with <id>.label.nii.gz labels. Samples are returned sorted by numeric id.
    """
    root = Path(data_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")

    logger.info("Scanning dataset under %s", root)
    patterns = ["*.img.nii.gz", "*.img.nii"]
    samples: Dict[int, Dict[str, object]] = {}

    for pattern in patterns:
        for img_path in sorted(root.rglob(pattern)):
            if not img_path.is_file():
                continue
            sample_id = _parse_numeric_id(img_path.name)
            if sample_id is None:
                logger.warning("Skipping file without numeric id: %s", img_path)
                continue
            if img_path.name.endswith(".img.nii.gz"):
                label_path = img_path.with_name(img_path.name.replace(".img.nii.gz", ".label.nii.gz"))
            else:
                label_path = img_path.with_name(img_path.name.replace(".img.nii", ".label.nii.gz"))
            if not label_path.exists():
                logger.warning("Label missing for id %s: %s (expected %s)", sample_id, img_path, label_path)
                continue
            if sample_id in samples and str(samples[sample_id]["image"]).endswith(".nii.gz"):
                # Prefer .nii.gz image when both .nii and .nii.gz are present
                continue
            samples[sample_id] = {
                "id": sample_id,
                "image": str(img_path.resolve()),
                "label": str(label_path.resolve()),
            }

    sorted_ids = sorted(samples.keys())
    return [samples[i] for i in sorted_ids]


def _save_split_file(split_path: Path, train: List[Dict], val: List[Dict], test: List[Dict], logger: logging.Logger) -> None:
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump({"train": train, "val": val, "test": test}, f, indent=2)
    logger.info("Saved split to %s", split_path)


def _map_split_entries(entries: Sequence, sample_index: Dict[int, Dict], split_name: str, logger: logging.Logger):
    mapped: List[Dict] = []
    missing: List[int] = []
    for entry in entries:
        sid = _parse_numeric_id(entry)
        if sid is None:
            logger.warning("Could not parse id from %s entry: %s", split_name, entry)
            continue
        sample = sample_index.get(sid)
        if sample:
            mapped.append(sample)
        else:
            missing.append(sid)
    mapped = sorted(mapped, key=lambda x: _parse_numeric_id(x.get("id")) or 1e9)
    return mapped, missing


def load_or_create_splits(
    samples: List[Dict],
    split_path: Path,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    logger: logging.Logger,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    split_path = Path(split_path)
    sample_index: Dict[int, Dict] = {}
    for s in samples:
        sid = _parse_numeric_id(s.get("id"))
        if sid is not None:
            sample_index[sid] = s

    if split_path.exists():
        try:
            with open(split_path) as f:
                split_data = json.load(f)
            train_loaded = split_data.get("train", [])
            val_loaded = split_data.get("val", [])
            test_loaded = split_data.get("test", [])
            train, miss_train = _map_split_entries(train_loaded, sample_index, "train", logger)
            val, miss_val = _map_split_entries(val_loaded, sample_index, "val", logger)
            test, miss_test = _map_split_entries(test_loaded, sample_index, "test", logger)
            missing_total = sorted(set(miss_train + miss_val + miss_test))
            if missing_total:
                logger.warning(
                    "Splits file at %s references %d id(s) not found in dataset (showing up to 10): %s",
                    split_path,
                    len(missing_total),
                    missing_total[:10],
                )
            if train or val or test:
                logger.info(
                    "Using existing split from %s (train=%d, val=%d, test=%d)",
                    split_path,
                    len(train),
                    len(val),
                    len(test),
                )
                return train, val, test
            logger.warning("Split file %s found but could not map any entries; creating a fresh split.", split_path)
        except Exception as exc:
            logger.warning("Failed to read existing split file %s: %s. Recreating splits.", split_path, exc)

    rng = random.Random(seed)
    items = samples.copy()
    rng.shuffle(items)
    n = len(items)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    _sort_key = lambda x: _parse_numeric_id(x.get("id")) if isinstance(x, dict) else None
    test = sorted(items[:n_test], key=lambda x: _sort_key(x) or 1e9) if n_test > 0 else []
    val = sorted(items[n_test:n_test + n_val], key=lambda x: _sort_key(x) or 1e9) if n_val > 0 else []
    train = sorted(items[n_test + n_val:], key=lambda x: _sort_key(x) or 1e9)

    _save_split_file(split_path, train, val, test, logger)
    logger.info("Created random split with seed %d: train=%d val=%d test=%d", seed, len(train), len(val), len(test))
    return train, val, test


def log_dataset_preview(samples: List[Dict], logger: logging.Logger) -> None:
    "Quick sanity logging on discovered samples and one loaded volume."
    logger.info("Total samples found: %d", len(samples))
    for sample in samples[:3]:
        logger.info("  id=%s image=%s label=%s", sample.get("id"), sample.get("image"), sample.get("label"))
    if not samples:
        return
    sample = samples[0]
    img_path = Path(sample["image"])
    lbl_path = Path(sample["label"])
    try:
        img_nii = nib.load(str(img_path))
        lbl_nii = nib.load(str(lbl_path))
        img_data = np.asarray(img_nii.get_fdata())
        lbl_data = np.asarray(lbl_nii.get_fdata())
        logger.info(
            "Sanity sample id=%s -> image shape %s (min %.4f max %.4f), label shape %s (min %.4f max %.4f)",
            sample.get("id"),
            tuple(img_data.shape),
            float(img_data.min()),
            float(img_data.max()),
            tuple(lbl_data.shape),
            float(lbl_data.min()),
            float(lbl_data.max()),
        )
    except Exception as exc:
        logger.warning("Failed to load sanity sample (%s, %s): %s", img_path, lbl_path, exc)


def normalize_case_id(raw: str) -> str:
    s = str(raw).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = re.sub(r"\.nii(\.gz)?$", "", s, flags=re.IGNORECASE)
    s = Path(s).stem
    return s


def load_imagecas_split(xlsx_path: Path, split_id: int, logger: logging.Logger) -> Tuple[Set[str], Set[str], Set[str]]:
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise ImportError("pandas (with openpyxl) is required to read the ImageCAS split file.") from exc

    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Split file not found: {xlsx_path}")

    def _read_with_header():
        try:
            return pd.read_excel(xlsx_path)
        except Exception:
            return None

    df = _read_with_header()
    if df is None or df.empty or df.shape[1] < 2:
        # try to auto-detect header row
        raw = pd.read_excel(xlsx_path, header=None)
        header_row = None
        for i in range(min(len(raw), 10)):
            row = [str(v).lower() for v in raw.iloc[i].tolist()]
            if any("split" in v for v in row) and any("file" in v or "id" in v for v in row):
                header_row = i
                break
        if header_row is not None:
            raw.columns = raw.iloc[header_row]
            df = raw.iloc[header_row + 1 :]
        else:
            df = raw

    cols_lower = [str(c).lower() for c in df.columns]
    id_col = None
    for c, lc in zip(df.columns, cols_lower):
        if "file" in lc or "id" in lc or "case" in lc:
            id_col = c
            break
    if id_col is None:
        id_col = df.columns[0]

    split_col_name = f"split-{split_id}"
    split_col = None
    for c, lc in zip(df.columns, cols_lower):
        if lc == split_col_name.lower():
            split_col = c
            break
    if split_col is None:
        for c in df.columns:
            if "split" in str(c).lower():
                split_col = c
                break
    if split_col is None:
        raise ValueError("Could not find a split column in the ImageCAS split file.")

    train_ids: Set[str] = set()
    val_ids: Set[str] = set()
    test_ids: Set[str] = set()
    for _, row in df.iterrows():
        rid = normalize_case_id(row[id_col])
        split_val = str(row[split_col]).strip().lower()
        if split_val in ("train", "training"):
            train_ids.add(rid)
        elif split_val in ("val", "valid", "validation"):
            val_ids.add(rid)
        elif split_val == "test":
            test_ids.add(rid)
    logger.info("Loaded ImageCAS split from %s (Split-%d): train=%d val=%d test=%d", xlsx_path, split_id, len(train_ids), len(val_ids), len(test_ids))
    return train_ids, val_ids, test_ids


# SECTION 3: DATA TRANSFORMS & AUGMENTATION

def binarize_label_foreground(x):
    """Convert multi-class HVSMR labels (0/1/2) or 0/255 masks to binary foreground."""
    try:
        return (x > 0).astype(np.float32)
    except AttributeError:
        return (x > 0).float()


def build_transforms(config: Config, mode: str = "train"):
    """Build MONAI transforms pipeline"""
    try:
        from monai.transforms import (
            Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
            Orientationd, Spacingd, ScaleIntensityRanged, NormalizeIntensityd,
            RandCropByPosNegLabeld, ResizeWithPadOrCropd, Resized,
            Rand3DElasticd, RandAdjustContrastd, RandGaussianNoised,
            RandGaussianSmoothd, RandFlipd, RandRotate90d, RandScaleIntensityd,
            SpatialPadd, ScaleIntensityRangePercentilesd, Lambdad,
        )
    except Exception as e:
        raise ImportError("monai is required for transforms. Install `pip install monai`") from e

    # Base preprocessing (all modes)
    base_transforms = [
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=config.pixdim,
            mode=("bilinear", "nearest")
        ),
    ]

    if config.modality == "ct":
        a_min, a_max = getattr(config, "ct_window", (-100, 700))
        base_transforms.append(
            ScaleIntensityRanged(
                keys=["image"],
                a_min=float(a_min),
                a_max=float(a_max),
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        )
    else:
        base_transforms.append(
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=0.5,
                upper=99.5,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        )
        base_transforms.append(
            NormalizeIntensityd(
                keys=["image"],
                nonzero=True,
                channel_wise=True
            )
        )

    augmentations = []
    if mode == "train":
        augmentations = [
            # pad to at least roi_size along each axis before cropping
            SpatialPadd(keys=["image", "label"], spatial_size=config.roi_size),
            # Oversample foreground to avoid all-zero crops on sparse vessels
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=config.roi_size,
                pos=2,
                neg=1,
                num_samples=4,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            Rand3DElasticd(
                keys=["image", "label"],
                prob=0.15,
                sigma_range=(4, 6),
                magnitude_range=(50, 150),
                mode=("bilinear", "nearest")
            ),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.2)),
            RandScaleIntensityd(keys=["image"], factors=(0.85, 1.15), prob=0.4),
            RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.2,
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5)
            ),
            # enforce consistent roi after spatial/intensity ops
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=config.roi_size),
        ]
    else:
        # For validation (Path C), use the same RandCropByPosNegLabeld as training.
        # This ensures validation patches have the same 2:1 foreground:background ratio
        # the model was trained on, allowing Dice and Hausdorff metrics to compute
        # on realistic (non-zero) predictions.
        # Note: These are "optimistic" validation metrics (on foreground-rich patches).
        # For realistic full-volume performance, use sliding-window inference on full volumes later.
        # For validation (Path C simple), don't use RandCropByPosNegLabeld.
        # Instead, validate on full resampled volumes. This avoids the issue where
        # RandCropByPosNegLabeld fails to find foreground. Metrics will be computed
        # on full volumes, which is more realistic than center-crop.
        augmentations = []

    # HVSMR labels are 0/1/2; binarize any foreground >0 before augmentation.
    label_transforms_early = [
        Lambdad(keys=["label"], func=binarize_label_foreground),
    ]
    # No additional label transforms needed after this
    label_transforms = []

    # Final type conversion
    final_transforms = [
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ]

    return Compose(base_transforms + label_transforms_early + augmentations + label_transforms + final_transforms)


# Model Architecture

def build_model(config: Config, logger: logging.Logger):
    "Build U-Net model with optional attention"
    try:
        from monai.networks.nets import UNet
    except Exception as e:
        raise ImportError("monai is required for model building. Install pip install monai") from e

    # simple 3D UNet
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=config.unet_channels,
        strides=config.unet_strides,
        num_res_units=config.unet_num_res_units,
        dropout=config.unet_dropout,
        norm=config.unet_norm,
    )

    model = model.to(config.device)

    if config.compile_model:
        if config.device == "mps":
            logger.warning("torch.compile is unstable on Apple MPS; skipping compilation.")
        elif not hasattr(torch, "compile"):
            logger.warning("torch.compile requested but torch.compile is unavailable in this PyTorch build.")
        else:
            try:
                model = torch.compile(model)  # type: ignore[attr-defined]
                logger.info("Enabled torch.compile for faster training.")
            except Exception as exc:  # pragma: no cover - runtime guard
                logger.warning(f"torch.compile failed ({exc}); proceeding without compilation.")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model parameters:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,}")
    logger.info(f"  Size: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")

    return model


# Loss Functions

def compute_class_weights(train_files: List[Dict], logger: logging.Logger, device: str) -> torch.Tensor:
    "Compute pos_weight for handling class imbalance"
    logger.info("Computing class weights from training data...")

    pos_vox = 0
    neg_vox = 0

    for item in tqdm(train_files, desc="Analyzing labels", unit="file"):
        label_path = Path(item["label"])
        try:
            nii = nib.load(str(label_path))
            data = np.asarray(nii.get_fdata())
            pos = int(np.count_nonzero(data > 0))
            neg = int(data.size - pos)
            pos_vox += pos
            neg_vox += neg
        except Exception as e:
            logger.warning(f"Could not read {label_path}: {e}")

    if pos_vox == 0:
        logger.warning("No positive voxels found in training labels. Using pos_weight=1.0")
        pos_vox = 1

    pos_weight_val = float(neg_vox / (pos_vox + 1e-9))
    pos_weight_val = min(pos_weight_val, 200.0)  # Cap to prevent instability

    logger.info(f"  Positive voxels: {int(pos_vox):,}")
    logger.info(f"  Negative voxels: {int(neg_vox):,}")
    logger.info(f"  Ratio (neg/pos): {pos_weight_val:.2f}")
    logger.info(f"  pos_weight = {pos_weight_val:.2f}")

    return torch.tensor([pos_weight_val], device=device)


def build_loss_function(pos_weight, logger: logging.Logger, use_focal: bool = False):
    "Hybrid loss for vascular segmentation (Dice+BCE or Dice+Focal for small vessels)"
    try:
        from monai.losses import DiceLoss, FocalLoss
    except Exception as e:
        raise ImportError("monai is required for losses. Install `pip install monai`") from e

    dice = DiceLoss(
        sigmoid=True,
        squared_pred=True,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
        reduction="mean",
    )

    if use_focal:
        bce_like = FocalLoss(
            to_onehot_y=False,
            gamma=2.0,
            alpha=0.25,
            reduction="mean",
        )
        bce_label = "Focal"
    else:
        bce_like = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        bce_label = "BCE"

    def combined_loss(logits, target):
        d = dice(logits, target)
        b = bce_like(logits, target)
        return 0.6 * d + 0.4 * b

    logger.info(f"Loss function: 60% Dice + 40% {bce_label} (sigmoid)")

    return combined_loss


# Training Model

class Trainer:
    "Main training engine"
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        if config.device.startswith("cuda"):
            self.use_amp = True
            self.autocast_device = "cuda"
        elif config.device == "mps":
            self.use_amp = False  # GradScaler/amp are not fully supported on MPS yet
            self.autocast_device = "mps"
        else:
            self.use_amp = False
            self.autocast_device = "cpu"
        self.non_blocking = config.device.startswith("cuda")
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.ckpt_best = None
        self.ckpt_last = None
        self.metrics_file = self.config.project_root / "training_metrics.json"
        self.best_loss = float("inf")
        self.use_tta = getattr(config, "use_tta", False)

    def mixup_3d(self, x1, y1, x2, y2, alpha=0.2):
        # basic mixup for augmentation
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        x = lam * x1 + (1 - lam) * x2
        y = lam * y1 + (1 - lam) * y2
        return x, y

    def _forward_with_tta(self, model, images):
        "Simple flip-based TTA for evaluation."
        # Added optional sliding window inference to support full-volume evaluation.
        # If `self.use_tta` is False, we still return `model(images)` for speed.
        if not self.use_tta:
            return model(images)
        logits = model(images)
        spatial_dims = [2, 3, 4]  # z, y, x for (B,C,Z,Y,X)
        for dims in ([spatial_dims[0]], [spatial_dims[1]], [spatial_dims[2]]):
            flipped = torch.flip(images, dims=dims)
            flipped_logits = torch.flip(model(flipped), dims=dims)
            logits = logits + flipped_logits
        return logits / 4.0

    def train_epoch(self, model, loader, optimizer, scaler, loss_fn, scheduler, epoch):
        model.train()
        running_loss = 0.0
        batches = 0
        accumulation_steps = max(1, self.config.accumulation_steps)
        effective_updates = 0
        last_raw_loss = None

        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
        for step, batch in enumerate(pbar, start=1):
            images = batch["image"].to(self.config.device, non_blocking=self.non_blocking)
            labels = batch["label"].to(self.config.device, non_blocking=self.non_blocking)

            with torch.amp.autocast(device_type=self.autocast_device, enabled=self.use_amp):
                logits = model(images)
                raw_loss = loss_fn(logits, labels)

            loss = raw_loss / accumulation_steps
            scaler.scale(loss).backward()

            raw_loss_value = float(raw_loss.item())
            running_loss += raw_loss_value
            batches += 1
            last_raw_loss = raw_loss_value

            if step % accumulation_steps == 0:
                if self.config.grad_clip_norm and self.config.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                effective_updates += 1

            pbar.set_postfix({
                "loss": f"{running_loss / max(1, batches):.4f}",
                "updates": effective_updates
            })

        remainder = batches % accumulation_steps
        if remainder != 0 and last_raw_loss is not None:
            if self.config.grad_clip_norm and self.config.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            effective_updates += 1

        avg = running_loss / max(1, batches)
        self.logger.info(f"Epoch {epoch} train loss: {avg:.4f}")
        return avg

    def validate_epoch(self, model, loader, loss_fn, metrics, epoch):
        if loader is None:
            self.logger.warning("Validation loader is None; skipping validation.")
            return float("inf"), {}
        try:
            loader_len = len(loader)
        except TypeError:
            loader_len = None
        if loader_len == 0:
            self.logger.warning("Validation loader is empty; skipping validation.")
            return float("inf"), {}

        model.eval()
        running_loss = 0.0
        n = 0
        metric_values = {}
        # fallback accumulators for robust metric computation (per-sample)
        fallback_dice_list = []
        fallback_haus_list = []
        # additional diagnostics: soft-dice (using probs) and probability stats
        soft_dice_list = []
        prob_mean_list = []
        prob_median_list = []

        if metrics:
            for metric in metrics.values():
                if hasattr(metric, "reset"):
                    metric.reset()

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Val Epoch {epoch}", leave=False):
                is_first_batch = (n == 0)
                images = batch["image"].to(self.config.device)
                labels = batch["label"].to(self.config.device)
                # DEBUG: Check raw label statistics
                if is_first_batch:
                    labels_np = labels[0, 0].cpu().numpy() if labels.shape[1] == 1 else labels[0].cpu().numpy()
                    self.logger.info(f"[DEBUG LABELS] Raw batch labels - shape: {labels.shape}, min: {labels_np.min():.6f}, max: {labels_np.max():.6f}, mean: {labels_np.mean():.6f}, count>0: {(labels_np > 0).sum()}")
                # Debug: log metadata for first batch to ensure correct pairing
                if is_first_batch:
                    try:
                        meta = batch.get("label_meta_dict", None)
                        if meta is None:
                            # sometimes meta stored under different key
                            meta = batch.get("label_meta", None)
                        self.logger.debug(f"First batch label meta: {meta}")
                    except Exception:
                        self.logger.debug("Could not read label meta for first batch")
                # Use sliding-window inference for validation to handle full-volume predictions
                # The model was trained on roi_size patches; use SWI to stitch overlapping predictions.
                try:
                    from monai.inferers import sliding_window_inference
                    logits = sliding_window_inference(
                        inputs=images,
                        roi_size=self.config.roi_size,
                        sw_batch_size=1,
                        predictor=lambda x: model(x),
                        overlap=0.5,
                        mode="gaussian"
                    )
                except Exception as e:
                    self.logger.warning(f"Sliding window inference failed: {e}; using standard forward")
                    logits = self._forward_with_tta(model, images)
                loss = loss_fn(logits, labels)
                running_loss += loss.item()
                n += 1

                # compute probabilities and binary predictions
                probs = torch.sigmoid(logits)
                preds = (probs > self.config.prob_threshold).float()

                # One-hot for metrics: channel 0 = background, channel 1 = vessel
                labels_bin = (labels > 0.5).float()
                preds_oh = torch.cat([1.0 - preds, preds], dim=1)
                labels_oh = torch.cat([1.0 - labels_bin, labels_bin], dim=1)
                pred_fg_sum = float(preds.sum().item())
                label_fg_sum = float(labels_bin.sum().item())

                # Per-batch/per-sample diagnostics: report label sums and prob means
                try:
                    for bi in range(preds.shape[0]):
                        if probs.shape[1] > 1:
                            p_prob = probs[bi, 1].cpu()
                        else:
                            p_prob = probs[bi, 0].cpu()
                        if labels_bin.shape[1] > 1:
                            g_float = labels_bin[bi, 1].cpu()
                        else:
                            g_float = labels_bin[bi, 0].cpu()
                        lsum = int(g_float.sum().item())
                        pmean = float(p_prob.mean().item())
                        self.logger.debug(f"Val batch {n} sample {bi} - label_sum: {lsum}, prob_mean: {pmean:.6f}")
                except Exception:
                    pass

                # Debug logging for first batch
                if is_first_batch:
                    self.logger.debug(f"First batch metric inputs - preds shape: {preds_oh.shape}, labels shape: {labels_oh.shape}")
                    self.logger.debug(f"Preds min/max: {preds_oh.min():.4f}/{preds_oh.max():.4f}, Labels min/max: {labels_oh.min():.4f}/{labels_oh.max():.4f}")
                    # Detailed logit/prob analysis
                    logits_np = logits[0, 0].cpu().detach().numpy()
                    probs_np = probs[0, 0].cpu().detach().numpy()
                    self.logger.info(f"[DIAGNOSTIC] Logits (ch 0) - min: {logits_np.min():.6f}, max: {logits_np.max():.6f}, mean: {logits_np.mean():.6f}, median: {np.median(logits_np):.6f}")
                    self.logger.info(f"[DIAGNOSTIC] Probs (ch 0) - min: {probs_np.min():.6f}, max: {probs_np.max():.6f}, mean: {probs_np.mean():.6f}, median: {np.median(probs_np):.6f}")
                    self.logger.info(f"[DIAGNOSTIC] Probs > 0.5: {(probs_np > 0.5).sum()} out of {probs_np.size} voxels")
                    self.logger.info(f"[DIAGNOSTIC] Probs > 0.1: {(probs_np > 0.1).sum()} out of {probs_np.size} voxels")
                    self.logger.info(f"[DIAGNOSTIC] Probs > 0.01: {(probs_np > 0.01).sum()} out of {probs_np.size} voxels")

                # Per-sample fallback Dice/Hausdorff (collected for robust reporting)
                bsize = preds.shape[0]
                for bi in range(bsize):
                    # Robustly determine foreground mask for predictions and labels.
                    # preds: (B,1,...) expected; labels_bin may be (B,1,...) or (B,2,...).
                    if preds.shape[1] > 1:
                        p_mask = preds[bi, 1].cpu().numpy().astype(bool)
                    else:
                        p_mask = preds[bi, 0].cpu().numpy().astype(bool)

                    if labels_bin.shape[1] == 1:
                        g_mask = labels_bin[bi, 0].cpu().numpy().astype(bool)
                    else:
                        # If labels are already one-hot with >=2 channels, take channel 1 as foreground.
                        # If uncertain, fallback to argmax.
                        try:
                            g_mask = labels_bin[bi, 1].cpu().numpy().astype(bool)
                        except Exception:
                            g_mask = (labels_bin[bi].argmax(axis=0).cpu().numpy() == 1)

                    p = p_mask
                    g = g_mask
                    # debug sums for first sample of first batch
                    if is_first_batch and bi == 0:
                        try:
                            self.logger.debug(f"Sample[0] sums - pred sum: {int(p.sum())}, label sum: {int(g.sum())}")
                        except Exception:
                            self.logger.debug(f"Sample[0] sums - pred sum (np): {p.sum()}, label sum (np): {g.sum()}")
                    inter = float((p & g).sum())
                    denom = float(p.sum() + g.sum())
                    dice_val = (2.0 * inter / denom) if denom > 0 else float('nan')
                    fallback_dice_list.append(dice_val)

                    # Soft-Dice computed from probabilities (more informative than hard threshold)
                    try:
                        if probs.shape[1] > 1:
                            p_prob = probs[bi, 1].cpu()
                        else:
                            p_prob = probs[bi, 0].cpu()
                        if labels_bin.shape[1] > 1:
                            g_float = labels_bin[bi, 1].cpu()
                        else:
                            g_float = labels_bin[bi, 0].cpu()
                        # ensure same shape
                        p_flat = p_prob.view(-1)
                        g_flat = g_float.view(-1)
                        soft_num = float((p_flat * g_flat).sum().item() * 2.0)
                        soft_den = float(p_flat.sum().item() + g_flat.sum().item() + 1e-6)
                        soft_d = (soft_num / soft_den) if soft_den > 0 else float('nan')
                        soft_dice_list.append(soft_d)
                        # probability stats
                        arr = p_flat.numpy()
                        prob_mean_list.append(float(arr.mean()))
                        prob_median_list.append(float(np.median(arr)))
                    except Exception:
                        soft_dice_list.append(float('nan'))
                        prob_mean_list.append(float('nan'))
                        prob_median_list.append(float('nan'))

                    # hausdorff fallback: prefer scipy.ndimage.distance_transform_edt
                    haus_val = float('nan')
                    try:
                        from scipy import ndimage
                        if g.sum() > 0 and p.sum() > 0:
                            dt_g = ndimage.distance_transform_edt(~g)
                            dt_p = ndimage.distance_transform_edt(~p)
                            d_p_to_g = dt_g[p]
                            d_g_to_p = dt_p[g]
                            if d_p_to_g.size and d_g_to_p.size:
                                all_d = np.concatenate([d_p_to_g.ravel(), d_g_to_p.ravel()])
                                haus_val = float(np.percentile(all_d, 95))
                    except Exception:
                        try:
                            from scipy.spatial import distance
                            p_idx = np.argwhere(p)
                            g_idx = np.argwhere(g)
                            if p_idx.size and g_idx.size:
                                if p_idx.shape[0] > 2000:
                                    choice = np.random.choice(p_idx.shape[0], 2000, replace=False)
                                    p_idx = p_idx[choice]
                                if g_idx.shape[0] > 2000:
                                    choice = np.random.choice(g_idx.shape[0], 2000, replace=False)
                                    g_idx = g_idx[choice]
                                d1 = distance.cdist(p_idx, g_idx).min(axis=1)
                                d2 = distance.cdist(g_idx, p_idx).min(axis=1)
                                haus_val = float(np.percentile(np.concatenate([d1, d2]), 95))
                        except Exception:
                            haus_val = float('nan')
                    fallback_haus_list.append(haus_val)

                # Update MONAI metrics if provided
                if metrics:
                    for name, metric in metrics.items():
                        try:
                            if name.lower().startswith("haus") and (pred_fg_sum == 0.0 or label_fg_sum == 0.0):
                                self.logger.debug(
                                    "Skipping Hausdorff: empty foreground (pred_sum=%.1f, label_sum=%.1f)",
                                    pred_fg_sum,
                                    label_fg_sum,
                                )
                                continue
                            metric(preds_oh, labels_oh)
                        except Exception as e:
                            self.logger.warning(f"Error updating metric {name}: {e}")

        val_loss = running_loss / max(1, n)
        self.logger.info(f"Epoch {epoch} val loss: {val_loss:.4f}")

        if metrics:
            self.logger.debug(f"Fallback collected counts before aggregation - dice: {len(fallback_dice_list)}, haus: {len(fallback_haus_list)}")
            for name, metric in metrics.items():
                try:
                    value = metric.aggregate()
                    self.logger.debug(f"Raw aggregation result for {name}: {value}, type: {type(value)}")
                except Exception as e:
                    self.logger.warning(f"Failed to aggregate metric {name}: {e}")
                    value = None
                
                not_nans = None
                if value is not None:
                    # Handle tuple return from MONAI metrics (value, not_nans)
                    if isinstance(value, tuple) and len(value) == 2:
                        value, not_nans = value
                        self.logger.debug(f"Extracted from tuple - value: {value}, not_nans: {not_nans}")
                    
                    # Convert tensor to scalar
                    if isinstance(value, torch.Tensor):
                        if value.numel() == 1:
                            value = value.item()
                        else:
                            value = value.mean().item()
                    elif isinstance(value, (list, tuple)):
                        if len(value) > 0:
                            value = float(torch.as_tensor(value).mean().item())
                        else:
                            value = None
                    
                    # If metric reports zero valid samples, treat as unavailable (n/a)
                    if not_nans is not None:
                        try:
                            if isinstance(not_nans, torch.Tensor):
                                not_nans_count = int(not_nans.sum().item())
                            else:
                                not_nans_count = int(not_nans) if isinstance(not_nans, (int, float)) else 0
                            if not_nans_count <= 0:
                                self.logger.debug(f"Metric {name} reports zero valid samples (not_nans={not_nans_count}); marking as n/a")
                                value = None
                        except Exception as e:
                            self.logger.warning(f"Error processing not_nans for {name}: {e}")
                    
                    if value is not None:
                        try:
                            metric_values[name] = float(value)
                            self.logger.debug(f"Metric {name} = {value}")
                        except (ValueError, TypeError) as e:
                            self.logger.warning(f"Could not convert metric {name} to float: {e}")
                # If MONAI could not produce a metric (value is None) use our fallback lists
                if value is None:
                    try:
                        if name.lower().startswith("dice") and len(fallback_dice_list) > 0:
                            v = float(np.nanmean(fallback_dice_list))
                            if not np.isnan(v):
                                metric_values[name] = v
                                self.logger.debug(f"Fallback dice (mean over samples) = {v}")
                        elif name.lower().startswith("haus") and len(fallback_haus_list) > 0:
                            v = float(np.nanmean(fallback_haus_list))
                            if not np.isnan(v):
                                metric_values[name] = v
                                self.logger.debug(f"Fallback hausdorff (mean over samples) = {v}")
                    except Exception as e:
                        self.logger.debug(f"Fallback metric computation failed for {name}: {e}")
                
                if hasattr(metric, "reset"):
                    metric.reset()
            
            # Display metrics with better error handling
            summary_parts = []
            for name in metrics.keys():
                v = metric_values.get(name)
                if v is not None:
                    summary_parts.append(f"{name}: {v:.4f}")
                else:
                    summary_parts.append(f"{name}: n/a")
            # Append diagnostic aggregated metrics (soft-dice and probability stats)
            try:
                if len(soft_dice_list) > 0:
                    sd = float(np.nanmean(soft_dice_list))
                    summary_parts.append(f"soft_dice: {sd:.4f}")
                    metric_values.setdefault('soft_dice', sd)
                if len(prob_mean_list) > 0:
                    pm = float(np.nanmean(prob_mean_list))
                    pmed = float(np.nanmean(prob_median_list))
                    summary_parts.append(f"prob_mean: {pm:.4f}")
                    summary_parts.append(f"prob_median: {pmed:.4f}")
                    metric_values.setdefault('prob_mean', pm)
                    metric_values.setdefault('prob_median', pmed)
            except Exception:
                # best-effort diagnostics; ignore failures here
                pass

            metrics_summary = " | ".join(summary_parts)
            self.logger.info(f"Epoch {epoch} val metrics: {metrics_summary}")

        return val_loss, metric_values

    def save_checkpoint(self, epoch, model, optimizer, scheduler, is_best=False):
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_loss": self.best_loss,
        }
        p = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(state, str(p))
        self.ckpt_last = str(p)
        if is_best:
            bestp = self.config.checkpoint_dir / "checkpoint_best.pt"
            shutil.copyfile(str(p), str(bestp))
            self.ckpt_best = str(bestp)

    def _load_checkpoint_via_buffer(self, checkpoint_path: Path):
        """
        Read the checkpoint file into memory before invoking torch.load. This
        forces cloud-backed placeholder files (e.g., OneDrive) to hydrate fully.
        """
        with checkpoint_path.open("rb") as handle:
            data = handle.read()
        if not data:
            raise RuntimeError(f"Checkpoint appears empty: {checkpoint_path}")
        buffer = io.BytesIO(data)
        return torch.load(buffer, map_location=self.config.device)

    def _find_alternate_checkpoint(self, failed_path: Path) -> Optional[Path]:
        candidates = []
        best = self.config.checkpoint_dir / "checkpoint_best.pt"
        if best.exists() and best != failed_path:
            candidates.append(best)

        epochs = sorted(
            self.config.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for candidate in epochs:
            if candidate == failed_path:
                continue
            candidates.append(candidate)

        return candidates[0] if candidates else None

    def load_checkpoint(self, model, optimizer=None, scheduler=None, checkpoint_path=None):
        if checkpoint_path is None:
            return None
        try:
            ckpt = torch.load(str(checkpoint_path), map_location=self.config.device)
        except RuntimeError as err:
            msg = str(err).lower()
            if "failed finding central directory" not in msg:
                if "size mismatch" in msg or "missing key" in msg:
                    raise RuntimeError(
                        f"Checkpoint {checkpoint_path} does not match the current model architecture. "
                        "Ensure --unet_channels/--unet_strides/--unet_res_units match the checkpoint "
                        "you trained (defaults are 32,64,128,256,512)."
                    ) from err
                raise
            self.logger.warning(
                "Checkpoint %s could not be loaded directly (%s); retrying via in-memory buffer.",
                checkpoint_path,
                err,
            )
            try:
                ckpt = self._load_checkpoint_via_buffer(Path(checkpoint_path))
            except RuntimeError as buffer_err:
                fallback = self._find_alternate_checkpoint(Path(checkpoint_path))
                if fallback:
                    self.logger.warning(
                        "Checkpoint %s appears corrupted; falling back to %s.",
                        checkpoint_path,
                        fallback,
                    )
                    return self.load_checkpoint(
                        model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        checkpoint_path=fallback,
                    )
                raise RuntimeError(
                    f"Checkpoint {checkpoint_path} is corrupted or incomplete. "
                    "Ensure it is fully downloaded (OneDrive: 'Always keep on this device'), "
                    "or delete it and resume from another checkpoint."
                ) from buffer_err
        model.load_state_dict(ckpt["model_state"])
        if optimizer is not None and ckpt.get("optimizer_state"):
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if ckpt.get("best_loss") is not None:
            self.best_loss = ckpt["best_loss"]
        return ckpt

    def train(self, model, train_loader, val_loader, optimizer, scheduler, loss_fn, metrics, start_epoch=0, writer=None):
        history = {"train_loss": [], "val_loss": []}
        val_metrics_history = {k: [] for k in metrics.keys()} if metrics else None
        epochs = getattr(self.config, "epochs", 10)
        start = int(start_epoch)
        end_epoch = start + epochs
        for epoch in range(start + 1, end_epoch + 1):
            train_loss = self.train_epoch(model, train_loader, optimizer, self.scaler, loss_fn, scheduler, epoch)
            val_loss, val_metric_values = self.validate_epoch(model, val_loader, loss_fn, metrics, epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            if val_metrics_history is not None:
                for name in val_metrics_history.keys():
                    val_metrics_history[name].append(val_metric_values.get(name))
            if writer is not None:
                writer.add_scalar("loss/train", train_loss, epoch)
                writer.add_scalar("loss/val", val_loss, epoch)
                if val_metric_values:
                    for name, value in val_metric_values.items():
                        writer.add_scalar(f"val/{name}", value, epoch)
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            self.save_checkpoint(epoch, model, optimizer, scheduler, is_best=is_best)
        # persist history
        out_history = {
            "start_epoch": start,
            "epochs_ran": epochs,
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
        }
        if val_metrics_history is not None:
            out_history["val_metrics"] = val_metrics_history
        with open(self.metrics_file, "w") as f:
            json.dump(out_history, f, indent=2)
        return out_history

    def plot_training_curves(self, history):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            self.logger.warning("Could not plot training curves (matplotlib missing?): %s", exc)
            return

        # Allow history to be reloaded from disk if only file exists
        if history is None and self.metrics_file.exists():
            with open(self.metrics_file) as f:
                history = json.load(f)
        if not history:
            self.logger.warning("No history to plot.")
            return

        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        val_metrics = history.get("val_metrics", {})
        epochs = range(1, len(train_loss) + 1)

        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        axes[0].plot(epochs, train_loss, label="Train Loss")
        axes[0].plot(epochs, val_loss, label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        ax = axes[1]
        for name, values in val_metrics.items():
            ys = [float(v) if v is not None else float("nan") for v in values]
            ax.plot(epochs, ys, label=name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric")
        ax.set_title("Validation Metrics")
        ax.legend()
        ax.grid(True, alpha=0.3)

        output_path = self.config.log_dir / "training_curves.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        self.logger.info("Saved training curves to %s", output_path)


# PIPELINE

def _resolve_dataset_path(config: Config, args: argparse.Namespace, logger: logging.Logger) -> Path:
    "Infer dataset location when the user does not pass --data_dir."
    if args.data_dir:
        candidate = Path(args.data_dir)
        if not candidate.exists():
            raise FileNotFoundError(f"--data_dir path does not exist: {candidate}")
        return candidate

    processed = config.processed_dir
    if any(processed.rglob("*.nii*")):
        logger.info(f"Using processed dataset at {processed}")
        return processed

    def _first_zip(directory: Path) -> Optional[Path]:
        for z in sorted(directory.glob("*.zip")):
            if zipfile.is_zipfile(z):
                return z
        return None

    raw_dir = config.data_dir / "raw"
    if raw_dir.exists():
        if any(raw_dir.rglob("*.nii*")):
            logger.info(f"Using raw directory with NIfTI files at {raw_dir}")
            return raw_dir
        zip_candidate = _first_zip(raw_dir)
        if zip_candidate:
            logger.info(f"Using dataset zip at {zip_candidate}")
            return zip_candidate

    zip_candidate = _first_zip(config.data_dir)
    if zip_candidate:
        logger.info(f"Using dataset zip at {zip_candidate}")
        return zip_candidate

    zip_candidate = _first_zip(config.project_root)
    if zip_candidate:
        logger.info(f"Using dataset zip at {zip_candidate}")
        return zip_candidate

    raise FileNotFoundError(
        "Could not automatically locate a dataset. Provide the path explicitly with "
        "--data_dir path/to/data_or_zip."
    )


def verify_data_pipeline(config: Config, paired_data: List[Dict], logger: logging.Logger, n: int = 2) -> None:
    "Verify dataset loading and transform behavior on a few samples."
    try:
        from monai.data.utils import pad_list_data_collate
    except Exception as e:
        raise ImportError("monai is required for verification. Install `pip install monai`") from e

    train_tf = build_transforms(config, mode="train")
    val_tf = build_transforms(config, mode="val")
    items = paired_data[:max(0, n)]

    fail_reasons: List[str] = []

    def _tensor_stats(t):
        try:
            mn = float(t.min().item())
            mx = float(t.max().item())
            mean = float(t.float().mean().item())
            numel = int(t.numel())
            return mn, mx, mean, numel
        except Exception:
            arr = np.asarray(t)
            return float(arr.min()), float(arr.max()), float(arr.mean()), int(arr.size)

    label_binarizer = binarize_label_foreground

    for idx, item in enumerate(items, start=1):
        img_path = Path(item["image"])
        lbl_path = Path(item["label"])
        sample_id = item.get("id", img_path.stem)

        logger.info("")
        logger.info(f"[VERIFY] Sample {idx}/{len(items)} - id={sample_id}")
        logger.info(f"         image: {img_path}")
        logger.info(f"         label: {lbl_path}")

        try:
            img_nii = nib.load(str(img_path))
            lbl_nii = nib.load(str(lbl_path))
            img_data = np.asarray(img_nii.get_fdata())
            lbl_data = np.asarray(lbl_nii.get_fdata())
        except Exception as exc:
            fail_reasons.append(f"Could not load NIfTI for {sample_id}: {exc}")
            logger.warning(f"Failed to load sample {sample_id}: {exc}")
            continue

        aff_diff = float(np.max(np.abs(img_nii.affine - lbl_nii.affine)))
        logger.info(f"Raw shapes: image {img_data.shape}, label {lbl_data.shape}")
        logger.info(f"Affine max|diff|: {aff_diff:.6f}")
        logger.info(f"Voxel spacings: image {img_nii.header.get_zooms()}, label {lbl_nii.header.get_zooms()}")
        unique_vals = np.unique(lbl_data)
        if unique_vals.size <= 20:
            logger.info(f"Label unique values: {unique_vals}")
        else:
            logger.info(
                "Label stats: min=%.6f, max=%.6f, unique_count=%d, sample=%s",
                float(lbl_data.min()),
                float(lbl_data.max()),
                int(unique_vals.size),
                unique_vals[:10]
            )
        fg_count = int((lbl_data > 0).sum())
        logger.info(f"Foreground voxels (label>0): {fg_count}")
        mapped_lbl = label_binarizer(lbl_data)
        mapped_uniques = np.unique(mapped_lbl)
        logger.info(f"Label unique values (mapped): {mapped_uniques}")
        mapped_fg = int((mapped_lbl > 0).sum())
        logger.info(f"Foreground voxels (mapped): {mapped_fg}")
        if getattr(config, "dataset", "") == "imagecas":
            frac_fg = mapped_fg / max(1, mapped_lbl.size)
            logger.info(f"Foreground fraction (mapped): {frac_fg:.8f}")
        if aff_diff > 1e-3:
            fail_reasons.append(f"{sample_id}: affine mismatch > 1e-3 ({aff_diff:.4e})")

        def _log_patch_stats(name: str, out_obj, label_sums: List[float], shape_mismatch: List[str]):
            if isinstance(out_obj, list):
                logger.info(f"{name}: list with {len(out_obj)} patch(es)")
                for pi, patch in enumerate(out_obj):
                    img_t = patch["image"]
                    lbl_t = patch["label"]
                    im_shape = tuple(img_t.shape)
                    lb_shape = tuple(lbl_t.shape)
                    if im_shape != lb_shape:
                        shape_mismatch.append(f"{name} patch {pi}: image shape {im_shape} vs label shape {lb_shape}")
                    i_min, i_max, i_mean, _ = _tensor_stats(img_t)
                    l_min, l_max, _, l_numel = _tensor_stats(lbl_t)
                    l_sum = float(lbl_t.sum().item()) if hasattr(lbl_t, "sum") else float(np.sum(lbl_t))
                    label_sums.append(l_sum)
                    frac_fg = l_sum / max(1, l_numel)
                    logger.info(
                        f"{name} patch {pi}: image shape {im_shape}, label shape {lb_shape}, "
                        f"image min/mean/max: {i_min:.4f}/{i_mean:.4f}/{i_max:.4f}, "
                        f"label min/max/sum: {l_min:.4f}/{l_max:.4f}/{l_sum:.2f}, "
                        f"label fg fraction: {frac_fg:.6f}"
                    )
            else:
                img_t = out_obj["image"]
                lbl_t = out_obj["label"]
                im_shape = tuple(img_t.shape)
                lb_shape = tuple(lbl_t.shape)
                if im_shape != lb_shape:
                    shape_mismatch.append(f"{name}: image shape {im_shape} vs label shape {lb_shape}")
                i_min, i_max, i_mean, _ = _tensor_stats(img_t)
                l_min, l_max, _, l_numel = _tensor_stats(lbl_t)
                l_sum = float(lbl_t.sum().item()) if hasattr(lbl_t, "sum") else float(np.sum(lbl_t))
                label_sums.append(l_sum)
                frac_fg = l_sum / max(1, l_numel)
                logger.info(
                    f"{name}: image shape {im_shape}, label shape {lb_shape}, "
                    f"image min/mean/max: {i_min:.4f}/{i_mean:.4f}/{i_max:.4f}, "
                    f"label min/max/sum: {l_min:.4f}/{l_max:.4f}/{l_sum:.2f}, "
                    f"label fg fraction: {frac_fg:.6f}"
                )

        # Train transforms
        out_train = train_tf({"image": str(img_path), "label": str(lbl_path)})
        train_label_sums: List[float] = []
        shape_mismatch_notes: List[str] = []
        _log_patch_stats("Train", out_train, train_label_sums, shape_mismatch_notes)

        if isinstance(out_train, list) and len(out_train) > 0:
            try:
                batch = pad_list_data_collate(out_train)
                logger.info(
                    "Collate (train) batch shapes - image: %s, label: %s",
                    tuple(batch["image"].shape),
                    tuple(batch["label"].shape),
                )
            except Exception as exc:
                fail_reasons.append(f"{sample_id}: collate failed ({exc})")
                logger.warning(f"Collate failed for {sample_id}: {exc}")

        # Validation transforms
        out_val = val_tf({"image": str(img_path), "label": str(lbl_path)})
        val_label_sums: List[float] = []
        _log_patch_stats("Val", out_val, val_label_sums, shape_mismatch_notes)

        if shape_mismatch_notes:
            fail_reasons.extend(f"{sample_id}: {msg}" for msg in shape_mismatch_notes)
        if train_label_sums and all(s == 0 for s in train_label_sums):
            fail_reasons.append(f"{sample_id}: labels appear empty after train transforms")
        if val_label_sums and all(s == 0 for s in val_label_sums):
            fail_reasons.append(f"{sample_id}: labels appear empty after val transforms")

    if fail_reasons:
        logger.warning("FAIL: data/transform verification found issues:")
        for msg in fail_reasons:
            logger.warning(f"  - {msg}")
    else:
        logger.info("PASS: data/transform verification completed without detected issues.")


def main(args):
    "Execute complete training pipeline"

    # Initialize configuration
    config = Config.from_args(args)
    config.epochs = args.epochs

    # Setup logging
    experiment_name = args.experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(config.log_dir, experiment_name)

    logger.info("Starting vascular segmentation training pipeline")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Device: {config.device}")
    logger.info("Dataset preset=%s dataset=%s", getattr(config, "dataset_preset", "custom"), getattr(config, "dataset", "custom"))
    if getattr(config, "dataset", "") == "imagecas":
        logger.info("ImageCAS preset active: modality=ct, window=%s, pixdim=%s, roi_size=%s, batch_size=%s, prob_threshold=%.2f",
                    getattr(config, "ct_window", None), config.pixdim, config.roi_size, config.batch_size, config.prob_threshold)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    if config.device == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None:
            logger.info(
                "MPS status - built: %s, available: %s",
                torch.backends.mps.is_built(),
                torch.backends.mps.is_available(),
            )
    if config.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = not torch.backends.cudnn.deterministic
        try:
            torch.set_float32_matmul_precision("medium")
        except AttributeError:
            logger.debug("torch.set_float32_matmul_precision unavailable; using default precision.")

    logger.info(f"Model compilation (torch.compile): {config.compile_model}")
    logger.info(
        "DataLoader workers: %s (prefetch=%s, persistent=%s)",
        config.num_workers,
        config.prefetch_factor if config.num_workers > 0 else "n/a",
        config.persistent_workers and config.num_workers > 0,
    )

    # Save configuration
    config.save(config.checkpoint_dir / "config.json")

    # Data Management
    logger.info("")
    logger.info("1. dataset discovery")

    data_manager = LocalDataManager(config, logger)
    data_root = Path(args.data_root) if args.data_root else (Path(args.data_dir) if args.data_dir else config.data_root)
    data_root = data_root.expanduser().resolve()
    if data_root.is_file() and zipfile.is_zipfile(data_root):
        data_root = data_manager.extract_dataset(data_root)
    samples = discover_cloud_samples(data_root, logger)
    if not samples:
        raise ValueError(f"No samples found under {data_root}. Expecting <id>.img.nii[.gz] with matching .label.nii.gz.")
    log_dataset_preview(samples, logger)
    if getattr(args, "verify_data", False):
        verify_data_pipeline(config, samples, logger, n=args.verify_n)
        return

    # Splitting
    logger.info("")
    logger.info("2. dataset split")

    splitter = DatasetSplitter(config, logger)
    use_official = getattr(args, "use_official_split", False) and config.dataset == "imagecas"
    if use_official:
        split_path = Path(args.split_xlsx) if args.split_xlsx else None
        if split_path is None and args.data_dir:
            candidate = Path(args.data_dir) / "imageCAS_data_split.xlsx"
            if candidate.exists():
                split_path = candidate
        if split_path is None:
            candidate = data_root / "imageCAS_data_split.xlsx"
            if candidate.exists():
                split_path = candidate
        if split_path is None:
            raise FileNotFoundError("Official split requested but imageCAS_data_split.xlsx not found. Pass --split_xlsx.")
        train_ids, val_ids, test_ids = load_imagecas_split(split_path, args.split_id, logger)
        paired_norm = []
        for it in samples:
            it_norm = dict(it)
            it_norm["id"] = normalize_case_id(it.get("id", Path(it["image"]).stem))
            paired_norm.append(it_norm)
        train_files = [p for p in paired_norm if p["id"] in train_ids]
        val_files = [p for p in paired_norm if p["id"] in val_ids]
        test_files = [p for p in paired_norm if p["id"] in test_ids]
        paired_ids = {p["id"] for p in paired_norm}
        excel_ids = train_ids | val_ids | test_ids
        missing_from_paired = sorted(list(excel_ids - paired_ids))[:10]
        missing_from_excel = sorted(list(paired_ids - excel_ids))[:10]
        logger.info("Official split counts: train=%d val=%d test=%d", len(train_files), len(val_files), len(test_files))
        logger.info("Sample train ids: %s", [p["id"] for p in train_files[:3]])
        logger.info("Sample val ids: %s", [p["id"] for p in val_files[:3]])
        logger.info("Sample test ids: %s", [p["id"] for p in test_files[:3]])
        if missing_from_paired:
            logger.warning("Excel IDs not found in paired data (first 10): %s", missing_from_paired)
        if missing_from_excel:
            logger.warning("Paired IDs not present in excel (first 10): %s", missing_from_excel)
        splitter._save_split(train_files, val_files, test_files)
    else:
        train_files, val_files, test_files = load_or_create_splits(
            samples,
            config.split_file,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            logger=logger,
        )
    if not train_files:
        msg = "The dataset split resulted in 0 training samples. "
        if use_official:
            msg += ("Official split IDs may not match discovered pairs. "
                    f"Example paired ids: {[p.get('id') for p in samples[:10]]}")
        else:
            msg += (f"Total paired items: {len(samples)}. "
                    "Add more labeled data or reduce the validation/test ratios.")
        raise ValueError(msg)
    if not val_files:
        logger.warning(
            "Validation split is empty. Adjust `--val_ratio` if you need validation metrics."
        )

    # Build Data Loaders
    logger.info("")
    logger.info("3. building data loaders")

    try:
        from monai.data import CacheDataset, DataLoader
        from monai.data.utils import pad_list_data_collate
    except Exception as e:
        raise ImportError("monai is required for dataloaders. Install `pip install monai`") from e

    train_transforms = build_transforms(config, mode="train")
    val_transforms = build_transforms(config, mode="val")

    logger.info("Creating training dataset (cache_rate=%.2f)...", config.cache_rate_train)
    t0 = time.perf_counter()
    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=config.cache_rate_train,
        num_workers=config.num_workers
    )
    logger.info("Training dataset ready in %.1fs", time.perf_counter() - t0)

    logger.info("Creating validation dataset (cache_rate=%.2f)...", config.cache_rate_val)
    t0 = time.perf_counter()
    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=config.cache_rate_val,
        num_workers=config.num_workers
    )
    logger.info("Validation dataset ready in %.1fs", time.perf_counter() - t0)

    train_loader_kwargs = {
        "batch_size": config.batch_size,
        "shuffle": True,
        "num_workers": config.num_workers,
        "collate_fn": pad_list_data_collate,
        "pin_memory": config.pin_memory,
    }
    # macOS DataLoader uses spawn, so avoid top-level side effects that run per worker.
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    train_loader_kwargs["generator"] = generator
    if config.num_workers > 0:
        if config.prefetch_factor:
            train_loader_kwargs["prefetch_factor"] = config.prefetch_factor
        train_loader_kwargs["persistent_workers"] = config.persistent_workers
        train_loader_kwargs["worker_init_fn"] = make_worker_init_fn(args.seed)
    train_loader = DataLoader(train_ds, **train_loader_kwargs)

    val_num_workers = max(1, config.num_workers // 2) if config.num_workers > 0 else 0
    val_loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": val_num_workers,
        "collate_fn": pad_list_data_collate,
        "pin_memory": config.pin_memory,
    }
    if val_num_workers > 0:
        if config.prefetch_factor:
            val_loader_kwargs["prefetch_factor"] = max(1, config.prefetch_factor // 2)
        val_loader_kwargs["persistent_workers"] = config.persistent_workers
        val_loader_kwargs["worker_init_fn"] = make_worker_init_fn(args.seed + 17)
    val_loader = DataLoader(val_ds, **val_loader_kwargs)

    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")

    # Build Model
    logger.info("")
    logger.info("4. model initialization")

    model = build_model(config, logger)

    # Loss and Metrics
    logger.info("")
    logger.info("5. loss function and metrics")

    pos_weight = compute_class_weights(train_files, logger, config.device)
    loss_fn = build_loss_function(pos_weight, logger, use_focal=args.use_focal_loss)

    # Metrics
    try:
        from monai.metrics import DiceMetric, HausdorffDistanceMetric
        metrics = {
            "dice": DiceMetric(
                include_background=False,  # focus on foreground channel
                reduction="mean",
                get_not_nans=True,
                ignore_empty=True,  # skip empty volumes to avoid artificial perfect scores
            ),
            "hausdorff": HausdorffDistanceMetric(
                include_background=False,
                percentile=95,
                reduction="mean",
                get_not_nans=True,
                directed=False,
            ),
        }
        logger.info("Metrics: Dice Score, Hausdorff Distance (95th percentile)")
    except Exception as e:
        metrics = {}
        logger.warning(
            "MONAI metrics not available; proceeding without Dice/Hausdorff tracking. Error: %s", e
        )

    # Optimizer & Scheduler
    logger.info("")
    logger.info("6. optimizer and scheduler")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )

    steps_per_epoch = max(1, len(train_loader) // max(1, config.accumulation_steps))

    t0_steps = max(1, config.scheduler_t0 * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=t0_steps,
        T_mult=config.scheduler_tmult,
        eta_min=config.scheduler_eta_min
    )

    logger.info(f"  Optimizer: AdamW")
    logger.info(f"  Initial LR: {config.learning_rate:.2e}")
    logger.info(
        f"  Scheduler: CosineAnnealingWarmRestarts (T0={t0_steps}, Tmult={config.scheduler_tmult}, eta_min={config.scheduler_eta_min})"
    )
    logger.info(f"  Steps per epoch: {steps_per_epoch}")

    # Training
    logger.info("")
    logger.info("7. training loop")

    trainer = Trainer(config, logger)
    resume_path = args.resume
    auto_resume_used = False
    if not resume_path and not getattr(args, "no_auto_resume", False):
        default_best = config.checkpoint_dir / "checkpoint_best.pt"
        if default_best.exists():
            resume_path = default_best
            auto_resume_used = True

    start_epoch = 0
    if resume_path:
        resume_path = Path(resume_path)
        if not resume_path.is_absolute():
            resume_path = config.checkpoint_dir / resume_path
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        if auto_resume_used:
            logger.info(f"Auto-resuming from best checkpoint: {resume_path}")
        else:
            logger.info(f"Resuming training from checkpoint: {resume_path}")
        try:
            ckpt = trainer.load_checkpoint(
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=resume_path
            )
        except RuntimeError as err:
            if auto_resume_used:
                logger.warning(
                    "Auto-resume failed because the checkpoint does not match the current model "
                    f"configuration: {err}\nStarting a fresh training run instead."
                )
                resume_path = None
            else:
                raise
        else:
            trainer.ckpt_last = str(resume_path)
            start_epoch = int(ckpt.get("epoch", 0))
            logger.info(
                f"Checkpoint loaded (epoch {start_epoch}). Continuing for {config.epochs} additional epochs."
            )

    if resume_path is None:
        logger.info("No checkpoint resume requested; starting from scratch.")

    tb_writer = None
    if getattr(args, "tensorboard", False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_log_dir = config.log_dir / "tensorboard" / experiment_name
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
            logger.info("TensorBoard logging enabled at %s", tb_log_dir)
        except Exception as exc:
            logger.warning("TensorBoard requested but unavailable: %s", exc)

    history = trainer.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        metrics=metrics,
        start_epoch=start_epoch,
        writer=tb_writer
    )

    # Matplotlib plot (enabled by default, disable with --no_plot_metrics)
    if getattr(args, "plot_metrics", True):
        trainer.plot_training_curves(history)

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

    logger.info("")
    logger.info("Training pipeline complete.")
    logger.info(f"Best model: {trainer.ckpt_best}")
    logger.info(f"Last model: {trainer.ckpt_last}")
    logger.info(f"Training metrics: {trainer.metrics_file}")
    logger.info(f"Logs: {config.log_dir}")

    return history


if __name__ == "__main__":
    print("Hello from CNN2.py")
    parser = argparse.ArgumentParser(
        description="Vascular Segmentation Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root directory containing <id>.img.nii[.gz] and <id>.label.nii.gz files (default: repository root)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to dataset zip file (e.g., ./data/raw/orig.zip)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.0,
        help="Test set ratio"
    )
    parser.add_argument(
        "--verify_data",
        action="store_true",
        help="Run dataset/transforms verification and exit."
    )
    parser.add_argument(
        "--verify_n",
        type=int,
        default=2,
        help="Number of samples to verify."
    )
    parser.add_argument(
        "--modality",
        type=str,
        choices=["mri", "ct"],
        default="mri",
        help="Select preprocessing defaults for MRI vs CT."
    )
    parser.add_argument(
        "--dataset_preset",
        type=str,
        default="custom",
        choices=["custom", "imagecas"],
        help="Optional preset that tunes preprocessing + training defaults for a known dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="custom",
        choices=["custom", "imagecas"],
        help="Dataset name (used for pairing/splits)."
    )
    parser.add_argument(
        "--use_official_split",
        action="store_true",
        help="Use official ImageCAS Excel split when available."
    )
    parser.add_argument(
        "--split_xlsx",
        type=str,
        default=None,
        help="Path to imageCAS_data_split.xlsx (default: auto-detect in data_dir)."
    )
    parser.add_argument(
        "--split_id",
        type=int,
        default=1,
        help="Split column to use from Excel (Split-1..Split-4)."
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay coefficient for AdamW"
    )
    parser.add_argument(
        "--roi_size",
        type=str,
        default="96,192,192",
        help="ROI size for training (format: H,W,D)"
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm value (set <=0 to disable)"
    )
    parser.add_argument(
        "--unet_channels",
        type=str,
        default="32,64,128,256,512",
        help="Comma-separated channel sizes for UNet encoder/decoder"
    )
    parser.add_argument(
        "--unet_strides",
        type=str,
        default="2,2,2,2",
        help="Comma-separated strides between UNet levels"
    )
    parser.add_argument(
        "--unet_res_units",
        type=int,
        default=3,
        help="Number of residual units per UNet level"
    )
    parser.add_argument(
        "--unet_dropout",
        type=float,
        default=0.15,
        help="Dropout probability inside UNet residual blocks"
    )
    parser.add_argument(
        "--scheduler_t0",
        type=int,
        default=8,
        help="Base restart period multiplier for the cosine scheduler"
    )
    parser.add_argument(
        "--scheduler_tmult",
        type=int,
        default=2,
        help="Restart period multiplier for the cosine scheduler"
    )
    parser.add_argument(
        "--scheduler_eta_min",
        type=float,
        default=1e-6,
        help="Minimum learning rate for the cosine scheduler"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override DataLoader worker count (defaults to 0 on Windows, 4 otherwise)"
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=None,
        help="DataLoader prefetch factor (requires num_workers > 0)"
    )
    parser.add_argument(
        "--no_persistent_workers",
        dest="persistent_workers",
        action="store_false",
        help="Disable persistent DataLoader worker processes"
    )
    parser.add_argument(
        "--compile",
        dest="compile_model",
        action="store_true",
        help="Enable torch.compile for faster training (PyTorch 2.0+)"
    )
    parser.add_argument(
        "--cache_rate_train",
        type=float,
        default=None,
        help="Fraction of the training set to cache in memory (0.0-1.0)."
    )
    parser.add_argument(
        "--cache_rate_val",
        type=float,
        default=None,
        help="Fraction of the validation set to cache in memory (0.0-1.0)."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from (relative to checkpoints/ if not absolute)"
    )
    parser.add_argument(
        "--use_focal_loss",
        action="store_true",
        help="Use Focal loss instead of BCE in the hybrid Dice loss (helps with tiny vessels)"
    )
    parser.add_argument(
        "--use_tta",
        action="store_true",
        help="Enable simple flip test-time augmentation during validation for smoother metrics"
    )
    parser.add_argument(
        "--prob_threshold",
        type=float,
        default=None,
        help="Probability threshold for converting sigmoid outputs to binary predictions during validation."
    )
    parser.add_argument(
        "--no_auto_resume",
        action="store_true",
        help="Disable automatic resume from checkpoint_best.pt"
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Log losses/metrics to TensorBoard (writes to logs/tensorboard/<experiment_name>)"
    )
    parser.add_argument(
        "--plot_metrics",
        dest="plot_metrics",
        action="store_true",
        help="Save a PNG plot of losses and metrics after training (logs/training_curves.png)"
    )
    parser.add_argument(
        "--no_plot_metrics",
        dest="plot_metrics",
        action="store_false",
        help="Disable saving the PNG plot"
    )

    # Experiment arguments
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name for this experiment"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    parser.set_defaults(
        persistent_workers=True,
        compile_model=False,
        plot_metrics=True,
    )

    args = parser.parse_args()

    set_global_seed(args.seed)

    main(args)
