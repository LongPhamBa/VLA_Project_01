
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train OpenVLA-like head with LoRA/QLoRA on Colab (optimized I/O + CPU pipeline)
- Keeps your original functionalities:
  * Load pretrained OpenVLA (fallback to ResNet18+Text encoder if unavailable)
  * Freeze vision/language backbone; add small action head
  * Fine-tune adapters (LoRA/QLoRA)
  * Track validation loss + accuracy (if labels exist) / MAE (always)
- Optimizations:
  * Avoid re-opening .npz per sample (per-file LRU cache)
  * Optional LMDB format for fastest loading (build once, then train)
  * Tensor-native transforms (no PIL)
  * Pre-tokenize commands & cache
  * DataLoader tuned: pin_memory, prefetch_factor, more workers
  * AMP mixed precision, cudnn.benchmark

Usage examples
--------------
# Train from .npz directory directly (fast path with caching)
python train_openvla_lora_optimized.py \
  --data_dir /content/data_npz \
  --save_dir /content/out \
  --batch_size 128 --epochs 5 --num_workers 4

# Build LMDB then train from it (fastest)
python train_openvla_lora_optimized.py --build_lmdb --data_dir /content/data_npz --lmdb_path /content/data.lmdb
python train_openvla_lora_optimized.py --use_lmdb --lmdb_path /content/data.lmdb --save_dir /content/out

Notes
-----
- If action labels are not present in files, we derive a coarse label by binning the continuous action.
- For QLoRA, pass --qlora to quantize the base model with bitsandbytes (4-bit). Colab T4 works with bnb 4-bit.
"""

import os, glob, time, math, random, argparse, io
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import models
import torchvision.transforms.functional as TF

from transformers import AutoTokenizer, AutoModel

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

try:
    import lmdb  # optional, for LMDB fast path
    LMDB_AVAILABLE = True
except Exception:
    LMDB_AVAILABLE = False

# Optional QLoRA (4-bit) via bitsandbytes
try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False

# ------------------------- utils -------------------------

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class LRUFileCache:
    """Keep up to `capacity` opened/parsed episodes in RAM to avoid repeated disk I/O.
    Each entry stores a dict of arrays: images, commands, actions, [labels].
    """
    def __init__(self, capacity: int = 32):
        self.capacity = capacity
        self.cache: OrderedDict[str, Dict] = OrderedDict()

    def get_episode(self, path: str) -> Dict:
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        # load
        with np.load(path, allow_pickle=True) as ep:
            data = {}
            # try multiple key variants for robustness
            def get_any(keys, default=None):
                for k in keys:
                    if k in ep: return ep[k]
                return default
            imgs = get_any([
                "steps/observation/image_front", "images", "obs/images", "image_front"
            ])
            cmds = get_any([
                "steps/language_instruction", "commands", "language_instruction"
            ])
            acts = get_any([
                "steps/action", "actions_continuous", "action"
            ])
            labels = get_any([
                "steps/action_label", "action_labels"
            ])
            if imgs is None or cmds is None or acts is None:
                raise KeyError(f"Missing arrays in {path}")
            data = {
                "images": imgs,
                "commands": cmds,
                "actions": acts.astype(np.float32),
            }
            if labels is not None:
                data["labels"] = labels.astype(np.int64)
        # insert
        self.cache[path] = data
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return data

# Action binning for a coarse accuracy metric when labels are absent.
# Assumes actions are 2D: [steer, throttle]. Adjust thresholds as needed.

def bin_action(a: np.ndarray) -> int:
    steer, throttle = float(a[0]), float(a[1])
    # Simple 5-way bin: left, right, forward, brake/stop, idle
    if throttle > 0.2:
        if steer < -0.15: return 0  # left-forward
        if steer > 0.15:  return 1  # right-forward
        return 2  # forward
    if throttle < -0.2: return 3  # reverse/brake
    # idle/low throttle
    if steer < -0.2: return 0
    if steer > 0.2:  return 1
    return 4  # idle

# ------------------------- Datasets -------------------------

class NPZOptimizedDataset(Dataset):
    def __init__(self, files: List[str], tokenizer, image_size: int = 224, cache_capacity: int = 32):
        self.files = files
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.cache = LRUFileCache(capacity=cache_capacity)
        self.index: List[Tuple[int, int]] = []  # (file_idx, step_idx)

        # Build index and pre-tokenize commands per episode to avoid per-sample tokenizer cost.
        self._tok_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        for fi, fpath in enumerate(self.files):
            try:
                ep = self.cache.get_episode(fpath)
                n = len(ep["images"])  # steps per episode
                self.index.extend([(fi, j) for j in range(n)])
                # Pre-tokenize all commands in this episode once
                cmds = [str(c) for c in ep["commands"]]
                toks = self.tokenizer(cmds, return_tensors="pt", padding=True, truncation=True, max_length=32)
                for j in range(n):
                    self._tok_cache[(fi, j)] = (
                        toks["input_ids"][j], toks["attention_mask"][j]
                    )
            except Exception as e:
                print(f"[WARN] Skipping {fpath}: {e}")
        print(f"[INFO] Indexed {len(self.index)} frames from {len(self.files)} files (NPZOptimizedDataset).")

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _img_to_tensor(img_np: np.ndarray, size: int) -> torch.Tensor:
        # img_np: HxWxC, uint8 or float in [0,1]
        if img_np.dtype != np.uint8:
            img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        t = torch.from_numpy(img_np).permute(2, 0, 1).contiguous().float() / 255.0
        t = TF.resize(t, [size, size], antialias=True)
        # normalize to [-1,1]
        t = (t - 0.5) / 0.5
        return t

    def __getitem__(self, idx):
        fi, j = self.index[idx]
        ep = self.cache.get_episode(self.files[fi])
        img = ep["images"][j]
        act = ep["actions"][j].astype(np.float32)
        input_ids, attn = self._tok_cache[(fi, j)]
        # optional label
        if "labels" in ep:
            label = int(ep["labels"][j])
        else:
            label = bin_action(act)
        return {
            "pixel_values": self._img_to_tensor(img, self.image_size),
            "input_ids": input_ids,
            "attention_mask": attn,
            "actions": torch.from_numpy(act),
            "labels": torch.tensor(label, dtype=torch.long),
        }

class LmdbDataset(Dataset):
    def __init__(self, lmdb_path: str, tokenizer, image_size: int = 224):
        if not LMDB_AVAILABLE:
            raise RuntimeError("lmdb package not installed. pip install lmdb")
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=True, max_readers=2048)
        with self.env.begin() as txn:
            self.length = int(txn.get(b"length").decode("utf-8"))
        self.tokenizer = tokenizer
        self.image_size = image_size

    @staticmethod
    def _img_to_tensor(img_np: np.ndarray, size: int) -> torch.Tensor:
        if img_np.dtype != np.uint8:
            img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        t = torch.from_numpy(img_np).permute(2, 0, 1).contiguous().float() / 255.0
        t = TF.resize(t, [size, size], antialias=True)
        t = (t - 0.5) / 0.5
        return t

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        key = f"{idx:08d}".encode("utf-8")
        with self.env.begin() as txn:
            buf = txn.get(key)
        # record layout: npz bytes with arrays: image (HWC uint8), command (str), action (2 float32), label (int64)
        ep = np.load(io.BytesIO(buf), allow_pickle=True)
        img = ep["image"]
        cmd = str(ep["command"])  # type: ignore
        act = ep["action"].astype(np.float32)
        if "label" in ep:
            label = int(ep["label"])  # type: ignore
        else:
            label = bin_action(act)
        tok = self.tokenizer(cmd, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
        return {
            "pixel_values": self._img_to_tensor(img, self.image_size),
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "actions": torch.from_numpy(act),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# ------------------------- Model wrappers -------------------------

class FallbackMultimodal(nn.Module):
    def __init__(self, text_model_name="bert-base-uncased"):
        super().__init__()
        self.vision = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.vision.fc = nn.Identity()
        vision_out = 512
        self.text = AutoModel.from_pretrained(text_model_name)
        text_out = self.text.config.hidden_size
        # freeze backbones
        for p in self.vision.parameters(): p.requires_grad = False
        for p in self.text.parameters(): p.requires_grad = False
        self.fusion = nn.Sequential(
            nn.Linear(vision_out + text_out, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.action_head = nn.Linear(128, 2)
        self.class_head = nn.Linear(128, 5)  # for coarse accuracy metric

    def forward(self, pixel_values, input_ids, attention_mask):
        v = self.vision(pixel_values)
        t = self.text(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        h = self.fusion(torch.cat([v, t], dim=-1))
        return self.action_head(h), self.class_head(h)

class OpenVLALikeWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        hidden = getattr(base_model.config, "hidden_size", 768)
        # freeze backbone
        for p in self.base.parameters(): p.requires_grad = False
        self.mlp = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU())
        self.action_head = nn.Linear(128, 2)
        self.class_head = nn.Linear(128, 5)

    def forward(self, pixel_values, input_ids, attention_mask):
        out = self.base(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state[:, 0, :]
        h = self.mlp(h)
        return self.action_head(h), self.class_head(h)

# ------------------------- LoRA / QLoRA -------------------------

def apply_lora(model: nn.Module, r=8, alpha=16, dropout=0.05):
    if not PEFT_AVAILABLE:
        print("[WARN] peft not installed; skipping LoRA")
        return model, False
    # Heuristic: target linear layers in small heads/MLP to keep it simple and robust
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ("mlp" in name or "action_head" in name or "class_head" in name):
            target_modules.append(name)
    if not target_modules:
        print("[WARN] No target linear layers found for LoRA; skipping")
        return model, False
    cfg = LoraConfig(r=r, lora_alpha=alpha, target_modules=list(set(target_modules)),
                     lora_dropout=dropout, bias="none", task_type="SEQ_CLS")
    wrapped = get_peft_model(model, cfg)
    wrapped.print_trainable_parameters()
    return wrapped, True

# ------------------------- Training -------------------------

@dataclass
class TrainConfig:
    data_dir: str = ""
    save_dir: str = "out"
    model_name: str = "openvla/openvla-pretrain"
    text_fallback: str = "bert-base-uncased"
    val_split: float = 0.1
    batch_size: int = 128
    num_workers: int = 4
    image_size: int = 224
    epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 1e-4
    accum_steps: int = 1
    use_amp: bool = True
    seed: int = 42
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    recursive: bool = False
    cache_capacity: int = 32
    # LMDB options
    use_lmdb: bool = False
    lmdb_path: str = ""
    build_lmdb: bool = False
    # QLoRA option
    qlora: bool = False


def build_lmdb_from_npz(npz_dir: str, lmdb_path: str, recursive: bool = False):
    if not LMDB_AVAILABLE:
        raise RuntimeError("lmdb not installed. pip install lmdb")
    pattern = "**/*.npz" if recursive else "*.npz"
    files = sorted(glob.glob(os.path.join(npz_dir, pattern), recursive=recursive))
    if not files:
        raise FileNotFoundError(f"No .npz files in {npz_dir}")
    env = lmdb.open(lmdb_path, map_size=4 * 1024**4)  # 4TB virtual map
    idx = 0
    with env.begin(write=True) as txn:
        for f in files:
            with np.load(f, allow_pickle=True) as ep:
                imgs = ep.get("steps/observation/image_front", ep.get("images"))
                cmds = ep.get("steps/language_instruction", ep.get("commands"))
                acts = ep.get("steps/action", ep.get("actions_continuous")).astype(np.float32)
                labels = ep.get("steps/action_label", ep.get("action_labels"))
                n = len(imgs)
                for j in range(n):
                    rec = {
                        "image": imgs[j],
                        "command": str(cmds[j]),
                        "action": acts[j].astype(np.float32),
                    }
                    if labels is not None:
                        rec["label"] = int(labels[j])
                    bio = io.BytesIO()
                    np.savez_compressed(bio, **rec)
                    txn.put(f"{idx:08d}".encode("utf-8"), bio.getvalue())
                    idx += 1
        txn.put(b"length", str(idx).encode("utf-8"))
    env.sync(); env.close()
    print(f"[LMDB] Wrote {idx} records to {lmdb_path}")


def create_dataloaders(cfg: TrainConfig):
    # tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(cfg.text_fallback, use_fast=True)

    if cfg.use_lmdb:
        ds = LmdbDataset(cfg.lmdb_path, tokenizer, image_size=cfg.image_size)
    else:
        pattern = "**/*.npz" if cfg.recursive else "*.npz"
        files = sorted(glob.glob(os.path.join(cfg.data_dir, pattern), recursive=cfg.recursive))
        if not files:
            raise FileNotFoundError(f"No .npz files in {cfg.data_dir}")
        ds = NPZOptimizedDataset(files, tokenizer, image_size=cfg.image_size, cache_capacity=cfg.cache_capacity)

    val_len = max(1, int(len(ds) * cfg.val_split))
    train_len = len(ds) - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(cfg.seed))

    nw = cfg.num_workers if cfg.num_workers > 0 else os.cpu_count() or 2
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        prefetch_factor=2 if nw > 0 else None,
        persistent_workers=True if nw > 0 else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        prefetch_factor=2 if nw > 0 else None,
        persistent_workers=True if nw > 0 else False,
    )
    return train_loader, val_loader


def create_model(cfg: TrainConfig, device: torch.device):
    base = None
    if cfg.qlora and BNB_AVAILABLE:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        bnb_config = None

    try:
        # Depending on OpenVLA packaging, this may or may not exist in HF.
        base = AutoModel.from_pretrained(cfg.model_name, quantization_config=bnb_config) if bnb_config else AutoModel.from_pretrained(cfg.model_name)
        model = OpenVLALikeWrapper(base)
        print("[INFO] Loaded base model:", cfg.model_name)
    except Exception as e:
        print(f"[WARN] Could not load {cfg.model_name} ({e}); falling back to ResNet18+BERT.")
        model = FallbackMultimodal(cfg.text_fallback)

    model.to(device)
    return model


def metrics_from_batch(pred_actions, true_actions, pred_logits, labels):
    # Regression losses
    l1 = F.l1_loss(pred_actions, true_actions)
    l2 = F.mse_loss(pred_actions, true_actions)
    # Classification accuracy (coarse)
    acc = (pred_logits.argmax(dim=-1) == labels).float().mean()
    return l1.item(), l2.item(), acc.item()


def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    # Optionally build LMDB then exit
    if cfg.build_lmdb:
        if not cfg.lmdb_path:
            raise ValueError("--build_lmdb requires --lmdb_path")
        build_lmdb_from_npz(cfg.data_dir, cfg.lmdb_path, cfg.recursive)
        return

    train_loader, val_loader = create_dataloaders(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(cfg, device)

    # Apply LoRA to small heads/MLP
    model, lora_applied = apply_lora(model, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout)

    # Optimizer on trainable params only
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    os.makedirs(cfg.save_dir, exist_ok=True)
    best_val = float("inf")
    best_path = None

    def forward_pass(batch):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        pred_actions, pred_logits = model(pixel_values, input_ids, attention_mask)
        reg_loss = F.smooth_l1_loss(pred_actions, actions)
        cls_loss = F.cross_entropy(pred_logits, labels)
        loss = reg_loss + 0.2 * cls_loss  # small weight for coarse classification
        return loss, pred_actions, actions, pred_logits, labels

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, train_mae, train_mse, train_acc = 0.0, 0.0, 0.0, 0.0

        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                loss, p_act, t_act, p_log, labs = forward_pass(batch)
            scaler.scale(loss / cfg.accum_steps).backward()
            if step % cfg.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            # running metrics
            mae, mse, acc = metrics_from_batch(p_act.detach(), t_act, p_log.detach(), labs)
            train_loss += loss.item()
            train_mae += mae
            train_mse += mse
            train_acc += acc

        # Validation
        model.eval()
        val_loss, val_mae, val_mse, val_acc = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                    loss, p_act, t_act, p_log, labs = forward_pass(batch)
                mae, mse, acc = metrics_from_batch(p_act, t_act, p_log, labs)
                val_loss += loss.item()
                val_mae += mae
                val_mse += mse
                val_acc += acc

        n_train = len(train_loader)
        n_val = len(val_loader)
        train_loss /= max(1, n_train)
        train_mae  /= max(1, n_train)
        train_mse  /= max(1, n_train)
        train_acc  /= max(1, n_train)
        val_loss   /= max(1, n_val)
        val_mae    /= max(1, n_val)
        val_mse    /= max(1, n_val)
        val_acc    /= max(1, n_val)

        dt = time.time() - t0
        print(f"[Epoch {epoch}]\n"
              f"  Train: loss {train_loss:.4f} | MAE {train_mae:.4f} | MSE {train_mse:.4f} | Acc {train_acc:.3f}\n"
              f"  Valid: loss {val_loss:.4f}  | MAE {val_mae:.4f}  | MSE {val_mse:.4f}  | Acc {val_acc:.3f}  | Time {dt:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(cfg.save_dir, f"best_epoch{epoch:02d}_val{val_loss:.4f}.pt")
            torch.save(model.state_dict(), best_path)
            print(f"[INFO] Saved best model to {best_path}")

    print("[DONE] Best val loss:", best_val, "->", best_path)


# ------------------------- CLI -------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--data_dir", type=str, default="")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--use_lmdb", action="store_true")
    p.add_argument("--lmdb_path", type=str, default="")
    p.add_argument("--build_lmdb", action="store_true")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--cache_capacity", type=int, default=32)
    # model
    p.add_argument("--model_name", type=str, default="openvla/openvla-pretrain")
    p.add_argument("--text_fallback", type=str, default="bert-base-uncased")
    p.add_argument("--qlora", action="store_true")
    # train
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    args = p.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        model_name=args.model_name,
        text_fallback=args.text_fallback,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        accum_steps=args.accum_steps,
        use_amp=not args.no_amp,
        seed=args.seed,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        recursive=args.recursive,
        cache_capacity=args.cache_capacity,
        use_lmdb=args.use_lmdb,
        lmdb_path=args.lmdb_path,
        build_lmdb=args.build_lmdb,
        qlora=args.qlora,
    )

    if cfg.build_lmdb:
        build_lmdb_from_npz(cfg.data_dir, cfg.lmdb_path, cfg.recursive)
    else:
        train(cfg)
