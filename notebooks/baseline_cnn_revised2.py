# %% [markdown]
# # Baseline CNN Reproduction Revised (Scoliosis1K)
# 
# **Paper Reference:** Table 3, Row 1 (Accuracy: 45.5%, Sensitivity: 99.0%, Specificity: 27.0%).
# 
# **Paper Definition of Baseline CNN:**
# > *"A basic model trained with cross-entropy loss, excluding horizontal pooling and BNNeck."*
# 
# **Implementation Choices to Match Paper:**
# - **Backbone**: OpenGait `ResNet9` (channels `[64, 128, 256, 512]`, strides `[1, 2, 2, 1]`).
# - **Temporal Aggregation**: Temporal Max Pooling (TP) over frames.
# - **Head**: Global Average Pooling -> Linear(512, 256) -> Linear(256, 3). *No BNNeck, no scale=16.*
# - **Loss**: Cross-Entropy only. Label smoothing 0.1 (OpenGait default).
# - **Optimizer**: SGD, LR 0.1, Momentum 0.9, Weight Decay 0.0005. Milestones at 10k, 14k, 18k.
# - **Data Pipeline**: `TripletSampler` (8 identities x 8 sequences) over the 1:1:8 split from `Scoliosis1K_118.json`.
# 
# **Expected Results Note**: The test set is heavily skewed toward Positive (419 out of 749). Blind guessing "Positive" yields ~56% accuracy. The paper's Baseline CNN achieving 45.5% is expected for a CE-only model without triplet identity supervision. It is completely normal if your reproduced accuracy lands in the 40-55% range.

# %%
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import copy, json, pickle, random, time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ==========================================
# 1. SETTINGS
# ==========================================
DATASET_ROOT = Path("/mnt/home/probation-01/kelompok8/OpenGait-ScoNet/datasets/Scoliosis1K-sil-pkl") 
PARTITION_JSON = Path("/mnt/home/probation-01/kelompok8/OpenGait-ScoNet/datasets/Scoliosis1K/Scoliosis1K_118.json")

LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
NUM_CLASSES = 3
N_FRAMES = 30

# Paper Hyperparameters
TOTAL_ITERS = 20000
MILESTONES = [10000, 14000, 18000]
TRAIN_LR = 0.1
IDS_PER_BATCH = 8
SEQS_PER_ID = 8
LABEL_SMOOTHING = 0.1

USE_AMP = False 

SEED = 0
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE} | PyTorch: {torch.__version__}")

# ==========================================
# 2. DATA LOADING & PREPROCESSING (Tetap)
# ==========================================
def load_pkl(path):
    with open(path, "rb") as f:
        seq = np.asarray(pickle.load(f))
    if seq.ndim == 4 and seq.shape[1] == 1: 
        seq = seq[:, 0]
    elif seq.ndim == 4 and seq.shape[-1] == 1:
        seq = seq[..., 0]
    return seq

def cut_and_normalize(seq):
    cut = seq.shape[-1] // 64 * 10
    if cut > 0:
        seq = seq[..., cut:-cut]
    return seq.astype(np.float32) / 255.0

def random_clip(seq, n_frames=N_FRAMES):
    idx = np.random.choice(len(seq), n_frames, replace=len(seq) < n_frames)
    return seq[idx]

def train_clip(seq, n_frames=N_FRAMES):
    return cut_and_normalize(random_clip(seq, n_frames))

@lru_cache(maxsize=128)
def load_pkl_cached(path):
    return load_pkl(path)

def build_index(root, subject_ids):
    items = []
    for sid in subject_ids:
        folder = root / str(sid)
        if not folder.is_dir(): continue
        for pkl in sorted(folder.rglob("*.pkl")):
            names = [p.lower() for p in pkl.relative_to(root).parts[:-1]]
            class_id = next((LABEL_TO_ID[n] for n in names if n in LABEL_TO_ID), None)
            if class_id is not None:
                items.append((str(sid), class_id, pkl))
    return items

def make_triplet_batch_fn(items, ids_per_batch, seqs_per_id):
    by_subject = {}
    for sid, class_id, path in items:
        by_subject.setdefault(sid, []).append((class_id, path))
    
    subjects_by_class = {0: [], 1: [], 2: []}
    for sid, seqs in by_subject.items():
        cid = seqs[0][0] 
        subjects_by_class[cid].append(sid)
        
    index = {sid: i for i, sid in enumerate(by_subject.keys())}

    def get_batch():
        clips, ids, classes = [], [], []
        class_quota = {2: 3, 1: 3, 0: 2} 
        
        for cid, quota in class_quota.items():
            pool = subjects_by_class[cid]
            num_to_pick = min(quota, len(pool))
            
            if num_to_pick > 0:
                chosen_sids = random.sample(pool, num_to_pick)
                for sid in chosen_sids:
                    seq_pool = by_subject[sid]
                    if len(seq_pool) >= seqs_per_id:
                        picks = random.sample(seq_pool, seqs_per_id)
                    else:
                        picks = random.choices(seq_pool, k=seqs_per_id)
                        
                    for class_id, path in picks:
                        clips.append(train_clip(load_pkl_cached(path)))
                        ids.append(index[sid])
                        classes.append(class_id)
                        
        return np.stack(clips), np.array(ids), np.array(classes)
    return get_batch

# %%
# ==========================================
# 3. RESNET9 ARCHITECTURE
# ==========================================
# 8 conv layers + 1 linear head = "ResNet9".
# Used here as a feature extractor (512-ch out).
def conv_bn(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class ResNet9(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = conv_bn(in_channels, 64)              # spatial preserved
        self.conv2 = conv_bn(64, 128, stride=2)            # /2
        self.res1  = nn.Sequential(conv_bn(128, 128),
                                   conv_bn(128, 128))      # residual @ /2
        self.conv3 = conv_bn(128, 256, stride=2)           # /4
        self.conv4 = conv_bn(256, 512, stride=2)           # /8
        self.res2  = nn.Sequential(conv_bn(512, 512),
                                   conv_bn(512, 512))      # residual @ /8

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out       # skip: 128-ch, same spatial
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out       # skip: 512-ch, same spatial
        return out                        # [N, 512, H/8, W/8]


class FrameWise(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        n, c, s, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(n * s, c, h, w)
        out = self.net(x)
        _, c2, h2, w2 = out.shape
        return out.reshape(n, s, c2, h2, w2).permute(0, 2, 1, 3, 4)


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = FrameWise(ResNet9(in_channels=1))
        self.embedding = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, sils):
        x = sils.unsqueeze(1)          # [n, 1, s, h, w]
        x = self.backbone(x)           # [n, 512, s, h//8, w//8]
        x = x.max(dim=2)[0]            # temporal max pool -> [n, 512, h//8, w//8]
        x = x.mean(dim=(2, 3))         # global spatial average -> [n, 512]
        embed = self.embedding(x)      # [n, 256]
        logits = self.classifier(embed)
        return embed.unsqueeze(-1), logits.unsqueeze(-1)


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) and m.affine:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def make_model():
    model = BaselineCNN()
    model.apply(init_weights)
    return model.to(DEVICE)

# %%
# ==========================================
# 4. TRAINING LOOP (CE ONLY, FLOAT32)
# ==========================================
def compute_loss(logits, classes, label_smoothing=0.1):
    logits = logits.mean(-1)   # [n, 3, 1] -> [n, 3]
    ce = F.cross_entropy(logits, classes, label_smoothing=label_smoothing)
    return ce, ce.detach()


def save_checkpoint(path, model, optimizer, scheduler, it, history):
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "it":        it,
        "history":   history,
    }, path)


def train(model, get_batch, iters, lr=0.1, milestones=(),
          log_every=100, label_smoothing=0.1,
          ckpt_path=None, save_every=5000, resume=False):

    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(milestones), gamma=0.1)

    history  = {"ce": [], "acc": [], "lr": []}
    start_it = 1

    # optional resume
    if resume and ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_it = ckpt["it"] + 1
        history  = ckpt["history"]
        print(f"[resume] from iter {start_it-1}")

    accum_steps = 2
    t0 = time.time()

    for it in range(start_it, iters + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        sils, ids, classes = get_batch()
        sils    = torch.as_tensor(sils,    dtype=torch.float32, device=DEVICE)
        classes = torch.as_tensor(classes, dtype=torch.long,    device=DEVICE)

        # split dinamis, tidak hardcode chunk_size=32
        s_chunks  = torch.chunk(sils,    accum_steps, dim=0)
        c_chunks  = torch.chunk(classes, accum_steps, dim=0)

        total_loss_val, total_acc = 0.0, 0.0
        for sc, cc in zip(s_chunks, c_chunks):
            _, logits = model(sc)
            ce, ce_val = compute_loss(logits, cc, label_smoothing)
            (ce / accum_steps).backward()
            total_loss_val += ce_val.item()
            total_acc += (logits.mean(-1).argmax(1) == cc).float().mean().item() / accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["ce"].append(total_loss_val)
        history["acc"].append(total_acc)

        if it == start_it or it % log_every == 0 or it in milestones:
            elapsed = time.time() - t0
            eta = elapsed / (it - start_it + 1) * (iters - it)
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"iter {it:5d} | LR {lr_now:.2e} | CE {total_loss_val:.4f} "
                  f"| acc {total_acc:.2f} | ETA {eta/60:.1f}m")

        if ckpt_path and save_every and (it % save_every == 0 or it == iters):
            save_checkpoint(ckpt_path, model, optimizer, scheduler, it, history)

    return history

# %%
# ==========================================
# 5. EVALUATION & VISUALIZATION
# ==========================================
@torch.no_grad()
def predict(model, seq_uint8, max_frames=720):
    model.eval()
    seq_uint8 = seq_uint8[:max_frames]
    x = torch.from_numpy(cut_and_normalize(seq_uint8)).unsqueeze(0).to(DEVICE)
    _, logits = model(x)
    scores = logits.mean(-1)
    probs = torch.softmax(scores, dim=-1)[0].cpu().numpy()
    return int(probs.argmax()), probs

def confusion(model, items, max_frames=720):
    conf = np.zeros((3, 3), dtype=int)
    for _, class_id, path in items:
        pred, _ = predict(model, load_pkl_cached(path), max_frames)
        conf[class_id, pred] += 1
    return conf

def paper_metrics(conf):
    accuracy = conf.trace() / max(conf.sum(), 1)
    sensitivity = conf[0, 0] / max(conf[0].sum(), 1) 
    specificity = conf[1:, 1:].sum() / max(conf[1:].sum(), 1) 
    return accuracy, sensitivity, specificity

def plot_all(history, conf, acc, sen, spe):
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    axes[0].plot(history["ce"], color='tab:blue', alpha=0.8)
    axes[0].set_title("Training Loss (CE)")
    axes[0].set_xlabel("Iteration")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["acc"], color='tab:green', alpha=0.8)
    axes[1].set_title("Training Accuracy")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["lr"], color='tab:red')
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_xlabel("Iteration")
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    im = axes[3].imshow(conf, cmap="Blues")
    for i in range(3):
        for j in range(3):
            axes[3].text(j, i, conf[i, j], ha="center", va="center", 
                         color="white" if conf[i, j] > conf.max() / 2 else "black")
    axes[3].set_xticks([0, 1, 2])
    axes[3].set_yticks([0, 1, 2])
    axes[3].set_xticklabels(LABELS)
    axes[3].set_yticklabels(LABELS)
    axes[3].set_xlabel("Predicted")
    axes[3].set_ylabel("True")
    axes[3].set_title(f"Confusion Matrix\nAcc: {acc:.1%} | Sen: {sen:.1%} | Spe: {spe:.1%}")
    fig.colorbar(im, ax=axes[3], fraction=0.046)
    
    plt.tight_layout()
    plt.show()

def evaluate(model, items, history):
    conf = confusion(model, items)
    acc, sen, spe = paper_metrics(conf)
    print(f"\nAccuracy: {acc:.1%} | Sensitivity: {sen:.1%} | Specificity: {spe:.1%}")
    plot_all(history, conf, acc, sen, spe)

# %%
# ==========================================
# 6. EXECUTE TRAINING
# ==========================================
CKPT_PATH = "baseline_cnn2_f32_full.pt"
RUN_FULL = True

if RUN_FULL:
    with open(PARTITION_JSON, "r", encoding="utf-8") as f:
        partition = json.load(f)
        
    train_items = build_index(DATASET_ROOT, partition["TRAIN_SET"])
    test_items = build_index(DATASET_ROOT, partition["TEST_SET"])
    print(f"Train sequences: {len(train_items)} | Test sequences: {len(test_items)}")
    
    set_seed()
    model = make_model()
    get_batch = make_triplet_batch_fn(train_items, IDS_PER_BATCH, SEQS_PER_ID)
    
    print(f"\nStarting training: {TOTAL_ITERS} iters, 8x8 batch, LR {TRAIN_LR}, Float32")
    history = train(
        model, get_batch, iters=TOTAL_ITERS, lr=TRAIN_LR, 
        milestones=MILESTONES, log_every=100, 
        label_smoothing=LABEL_SMOOTHING, ckpt_path=CKPT_PATH, save_every=5000, resume=False
    )
    
    print("\nTraining Complete. Evaluating...")
    evaluate(model, test_items, history)


