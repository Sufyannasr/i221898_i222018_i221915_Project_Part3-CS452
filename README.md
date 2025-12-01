# HybridLungViT — README

HybridLungViT — a PyTorch implementation for pediatric chest X-ray classification (Normal vs Pneumonia) that fuses a DEiT vision transformer stem with lightweight transformer encoders and a segmentation auxiliary head. The repo / notebook trains with stage-wise freezing, uses an unsupervised heuristic lung-mask generator, supports simple TTA, and saves the best model by validation AUC.

# Summary

This script/notebook:

Loads the Kermany chest X-ray pediatric dataset (train and test directories).

Generates unsupervised lung masks via simple CLAHE + morphology heuristics (used to create left/right/global token partitions).

Uses a DEiT model (via timm) as a stem to extract patch features, projects them to d_model channels, and produces a per-pixel feature map.

Builds left/right/global token sequences from feature maps and processes them with small transformer encoders and a cross-attention module; classification is predicted from a learned CLS token.

Optionally uses a segmentation head (disabled by default: LAMBDA_SEG = 0.0) and computes auxiliary segmentation losses.

Trains with class-weighted cross entropy, optional label smoothing, mixed precision, and stage-wise freezing (stem frozen initially).

Uses a simple horizontal flip TTA for evaluation and saves the best model by validation AUC.

Repository layout / important files

(Adapt paths if you convert this notebook into a script or package.)

.
├─ train_notebook.ipynb      # or script containing the code you pasted
├─ README.md                 # <-- this file
├─ ckpt_hybrid_deit_fixed/   # checkpoint directory (created by code)
│  ├─ best_model.pth
│  ├─ interrupted_epochX_ckpt.pth
├─ /kaggle/input/xraydata/chest_xray/  # expected dataset root (Kaggle layout)

# Requirements & installation

Recommended Python environment (example):

## create venv
python -m venv venv
source venv/bin/activate

## install core libs
pip install torch torchvision timm numpy pandas scikit-learn matplotlib tqdm opencv-python pillow

Suggested versions (approx):

Python 3.8+

PyTorch 1.10+ (or newer)

torchvision compatible with your torch build

timm (0.6+)

opencv-python, numpy, pandas, scikit-learn, tqdm, matplotlib, Pillow

If you are running on Kaggle: you can directly add the Chest X-ray Images (Pneumonia) dataset from the notebook "Add data" UI. The code expects the dataset at DATA_ROOT = "/kaggle/input/xraydata/chest_xray" (modify DATA_ROOT at the top of the script/notebook if needed).

# Dataset

This code expects the Kermany pediatric chest X-ray dataset arranged like:

chest_xray/
  train/
    PNEUMONIA/  (jpg/png files)
    NORMAL/
  test/
    PNEUMONIA/
    NORMAL/


At startup the notebook runs a verify_dataset() check and will raise if directories are missing.

# How to run

If you're using the notebook: run cells top to bottom.

If you extracted into a script (e.g. train.py), you can run:

# CPU / GPU
python train.py


Common edits before running:

DATA_ROOT — path to dataset root

IMG_SIZE, BATCH_SIZE, EPOCHS, LR — change as desired

LAMBDA_SEG / LAMBDA_ATT — enable auxiliary losses if you want segmentation/attention regularization

SAVE_DIR — where checkpoints will be saved

To run only evaluation on saved best model (after saving best_model.pth):

model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth"), map_location=DEVICE))
evaluate(model, test_loader, detailed=True, use_tta=True)

# Configuration / key hyperparameters

Defaults found at top of the script:

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/kaggle/input/xraydata/chest_xray"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
LR = 5e-5
WEIGHT_DECAY = 1e-4
LAMBDA_SEG = 0.0
LAMBDA_ATT = 0.0
SAVE_DIR = "./ckpt_hybrid_deit_fixed"
PATIENCE = 5
freeze_epochs = 3


# Training schedule:

Stem (DEiT) frozen for first freeze_epochs epochs, then unfrozen and LR lowered to LR * 0.2.

Optimizer: AdamW

Scheduler: ReduceLROnPlateau (monitors val AUC)

Mixed precision via torch.cuda.amp.GradScaler

Loss components:

ce = CrossEntropyLoss(weight=class_weights, label_smoothing=0.1) (if PyTorch installed supports label smoothing)

seg_loss = 0.5 * BCE + 0.5 * Dice

Final loss: loss = ce + LAMBDA_SEG * seg_loss + LAMBDA_ATT * attn_reg

What the code does (high level)

# Data loading

PediatricCXRDataset loads images, constructs an unsupervised lung mask per image (via generate_lung_mask_orig) and returns (img_tensor, label, mask_tensor).

Augmentations: RandomResizedCrop, horizontal flip, color jitter, rotation during training; plain resize + normalize in validation/test.

Stem

DEiTStem uses timm.create_model('deit_base_patch16_224', pretrained=True, num_classes=0) and extracts patch tokens, reshapes to feature map (B, 768, 14, 14), projects to d_model channels and upsamples to (56,56) feature map.

Tokenization / Left-Right-Global

Feature map is interpolated to IMG_SIZE x IMG_SIZE and the lung mask (heuristic) determines a per-sample vertical cut (mean x of mask). Left / right submaps are pooled into token_grid tokens per side and processed by SimpleTransformerEncoder (small transformer encoder). A global set of tokens is also computed.

Fusion & Classification

Cross-attention (CrossAttentionFusion) between a learnable CLS token and kv (left+right tokens) produces a fused CLS representation. A classification head maps CLS to 2 logits.

Segmentation auxiliary

SegHead upsamples stem features to produce a 1-channel segmentation prediction (if LAMBDA_SEG > 0 this contributes to training loss).

# Evaluation

evaluate() uses TTA (tta_probs, simple horizontal flip) and reports accuracy, AUC (ROC AUC), seg loss mean, classification report and confusion matrix if detailed=True.

Outputs & checkpoints

Saved into SAVE_DIR (default ./ckpt_hybrid_deit_fixed):

best_model.pth — model state dict whenever validation AUC improves.

interrupted_epoch{E}_ckpt.pth — saved on keyboard interrupt; contains model_state, opt_state, scaler_state, and some logs.

Logs printed to console (epoch time, val metrics, scheduler messages, early stopping).

Evaluation / TTA / Metrics

Evaluation metrics: accuracy, ROC AUC (via roc_auc_score), segmentation loss (dice + BCE used during training), classification report and confusion matrix (if detailed=True).

TTA: The function tta_probs averages predictions for original image and horizontally flipped image (flip mask too). Use use_tta=True to enable during evaluate().

Tips, troubleshooting and tuning suggestions

OOM / GPU memory

Reduce BATCH_SIZE or IMG_SIZE (e.g., 160 or 128) if memory runs out.

Set num_workers=0 for debugging; num_workers=2..8 for speed on local/Colab/Kaggle.

If still OOM, freeze DEiT longer or decrease d_model / projection channels.

If training is too slow

Use fewer timm parameters (e.g., deit_tiny_patch16_224 or other lightweight models).

Reduce token_grid (fewer tokens -> fewer transformer computations).

To enable segmentation aux loss

Set LAMBDA_SEG = 0.1 (or smaller) to start, then tune. Segmentation target quality relies on the heuristic mask — it's noisy, so moderate weight is recommended.

Improve data balance or class handling

The code uses a WeightedRandomSampler and class weights in loss; you can also try focal loss or oversampling.

Resuming training

Load checkpoint: ckpt = torch.load(path); model.load_state_dict(ckpt['model_state']); opt.load_state_dict(ckpt['opt_state']); scaler.load_state_dict(ckpt['scaler_state']) and set start_epoch = ckpt['epoch'] + 1.

Debugging

If cv2.imread() returns None, check file extensions and paths. The dataset loader already skips non-image files.

If AUC is nan, ensure y_true contains both classes in the evaluation batch or compute AUC on the full set.

Common improvements to try

Stronger augmentations (RandAugment), CutMix, MixUp.

Replace heuristic lung masks with a trained lung segmentation model if available.

Larger transformer encoders or more fusion layers.

Focal loss / class specific sampling / temperature scaling.

Reproducibility notes

The script sets seeds for random, numpy, and torch. For determinism you may need to set:

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


but note that this can slow training and some operations remain nondeterministic on GPU.

# License & citations

You can use/modify this code under the terms you prefer. Example: add an MIT license file if you intend open source.

Suggested citations / acknowledgements:

DEiT: Touvron et al., Data-efficient Image Transformers, for the DEiT stem (used from timm).

Dataset: Kermany et al., [Chest X-ray Images (Pneumonia) dataset] (community / Kaggle).

timm library authors for the model zoo utilities.

(Include formal citations in papers you submit.)

Final notes

The segmentation branch is disabled by default (LAMBDA_SEG = 0.0). Start with classification-only training, then enable segmentation with a small weight to regularize the model.
