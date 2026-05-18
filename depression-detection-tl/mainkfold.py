import os
import argparse
import yaml
import wandb
import torch
import random
import numpy as np

from tqdm import tqdm
from torch.utils.data import ConcatDataset, Subset, DataLoader
from sklearn.model_selection import KFold

from utils.transfer import apply_transfer_learning

from models import TMeanNet, DepressionDetector, TAMFN
from models.chunk_cross_improved import ChunkCrossAttentionNet
from models.chunk_transformer import ChunkTransformerNet
from datasets import get_dvlog_dataloader
from datasets.dvlog import _collate_fn

CONFIG_PATH = "./config.yaml"


# =========================
# SEED
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# ARGUMENTS
# =========================
def parse_args():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--train_gender", type=str)
    parser.add_argument("--test_gender", type=str)

    parser.add_argument(
        "-m", "--model", type=str,
        choices=["TMeanNet", "DepressionDetector", "TAMFN",
                 "ChunkCrossAttentionNet", "ChunkTransformerNet"]
    )

    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-bs", "--batch_size", type=int)
    parser.add_argument("-lr", "--learning_rate", type=float)

    parser.add_argument(
        "-sch", "--lr_scheduler", type=str,
        choices=["cos", "None"]
    )

    parser.add_argument("-d", "--device", type=str, nargs="*")

    parser.add_argument("--seed", type=int)

    parser.add_argument("--num_folds", type=int, default=5)

    parser.set_defaults(**config)
    return parser.parse_args()


# =========================
# TRAIN
# =========================
def train_epoch(net, train_loader, loss_fn, optimizer, lr_scheduler,
                device, current_epoch, total_epochs):

    net.train()
    sample_count = 0
    running_loss = 0.
    correct_count = 0
    TP, FP, TN, FN = 0, 0, 0, 0

    with tqdm(train_loader, desc=f"Training {current_epoch}/{total_epochs}", leave=False) as pbar:
        for x, y in pbar:
            x, y = x.to(device), y.to(device).unsqueeze(1)

            y_pred = net(x)
            loss = loss_fn(y_pred, y.float())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            sample_count += x.shape[0]
            running_loss += loss.item() * x.shape[0]

            pred = (y_pred > 0).int()
            correct_count += (pred == y).sum().item()

            TP += ((pred == 1) & (y == 1)).sum().item()
            FP += ((pred == 1) & (y == 0)).sum().item()
            TN += ((pred == 0) & (y == 0)).sum().item()
            FN += ((pred == 0) & (y == 1)).sum().item()

            pbar.set_postfix({
                "loss": running_loss / sample_count,
                "acc": correct_count / sample_count,
            })

    if lr_scheduler is not None:
        lr_scheduler.step()

    return {
        "loss": running_loss / sample_count,
        "acc": correct_count / sample_count,
    }


# =========================
# VALIDATION
# =========================
def val(net, val_loader, loss_fn, device, print_cm=False):
    net.eval()

    sample_count = 0
    running_loss = 0.
    TP, FP, TN, FN = 0, 0, 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)

            y_pred = net(x)
            loss = loss_fn(y_pred, y.float())

            sample_count += x.shape[0]
            running_loss += loss.item() * x.shape[0]

            pred = (y_pred > 0).int()

            TP += ((pred == 1) & (y == 1)).sum().item()
            FP += ((pred == 1) & (y == 0)).sum().item()
            TN += ((pred == 0) & (y == 0)).sum().item()
            FN += ((pred == 0) & (y == 1)).sum().item()

    loss = running_loss / sample_count
    acc = (TP + TN) / sample_count

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    balanced_acc = (recall + specificity) / 2

    denom = ((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)) ** 0.5
    mcc = (TP * TN - FP * FN) / denom if denom > 0 else 0.0

    # weighted metrics
    N_pos = TP + FN
    N_neg = TN + FP
    N_total = N_pos + N_neg

    precision_neg = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    recall_neg = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1_neg = (2 * precision_neg * recall_neg / (precision_neg + recall_neg)) if (precision_neg + recall_neg) > 0 else 0.0

    weighted_precision = (N_pos * precision + N_neg * precision_neg) / N_total
    weighted_recall = (N_pos * recall + N_neg * recall_neg) / N_total
    weighted_f1 = (N_pos * f1 + N_neg * f1_neg) / N_total

    if print_cm:
        print("\n========== CONFUSION MATRIX ==========")
        print(f"Actual anxious       : {TP + FN}")
        print(f"Actual non-anxious   : {TN + FP}")
        print(f"Predicted anxious    : {TP + FP}")
        print(f"Predicted non-anxious: {TN + FN}")
        print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")

    return {
        "loss": loss,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_acc": balanced_acc,
        "mcc": mcc,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }


# =========================
# MODEL
# =========================
def build_model(args):
    if args.model == "TMeanNet":
        return TMeanNet(hidden_sizes=[512, 512, 512])

    elif args.model == "DepressionDetector":
        return DepressionDetector(d=256, l=6, t_downsample=4)

    elif args.model == "TAMFN":
        return TAMFN(d=256, l=6, t_downsample=4)

    elif args.model == "ChunkCrossAttentionNet":
        return ChunkCrossAttentionNet(d_model=256, chunk_size=20, num_heads=8, dropout=0.5, num_gru_layers=2)

    elif args.model == "ChunkTransformerNet":
        return ChunkTransformerNet(d_model=256, chunk_size=20, num_heads=8, num_encoder_layers=3, dropout=0.5)

    else:
        raise ValueError("Unknown model")


# =========================
# MAIN
# =========================
def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device[0]

    train_loader = get_dvlog_dataloader(args.data_dir, "train", args.batch_size, args.train_gender)
    val_loader = get_dvlog_dataloader(args.data_dir, "valid", args.batch_size, args.test_gender)
    test_loader = get_dvlog_dataloader(args.data_dir, "test", args.batch_size, args.test_gender)

    full_dataset = ConcatDataset([
        train_loader.dataset,
        val_loader.dataset,
        test_loader.dataset
    ])

    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):

        print(f"\n========== FOLD {fold+1}/{args.num_folds} ==========")

        # wandb.init(
        #     project="dvlog-kfold",
        #     entity="sambitsahoo-k-iiser-bhopal",
        #     config=args,
        #     name=f"{args.model}-fold{fold+1}"
        # )

        wandb.init(mode="disabled")

        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        train_loader_fold = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn =_collate_fn)
        val_loader_fold = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn =_collate_fn)

        net = build_model(args).to(device)
        net = apply_transfer_learning(net, args)

        if len(args.device) > 1:
            net = torch.nn.DataParallel(net, device_ids=args.device)

        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

        if args.lr_scheduler == "cos":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs // 5, eta_min=args.learning_rate / 20
            )
        else:
            lr_scheduler = None

        best_val_acc = -1
        best_metrics = None

        for epoch in range(args.epochs):
            train_results = train_epoch(net, train_loader_fold, loss_fn,
                                        optimizer, lr_scheduler, device,
                                        epoch, args.epochs)

            val_results = val(net, val_loader_fold, loss_fn, device)

            if val_results["acc"] > best_val_acc:
                best_val_acc = val_results["acc"]
                best_metrics = val_results

                
                dataset_name = os.path.basename(args.data_dir)
                save_dir = f"weights_kfold/{args.model}_{dataset_name}"
                os.makedirs(save_dir, exist_ok=True)

                save_path = f"{save_dir}/fold{fold+1}.pt"

                if isinstance(net, torch.nn.DataParallel):
                    torch.save(net.module.state_dict(), save_path)
                else:
                    torch.save(net.state_dict(), save_path)

            wandb.log({
                "train_loss": train_results["loss"],
                "val_loss": val_results["loss"],
                "val_acc": val_results["acc"],
                "val_f1": val_results["f1"]
            })

        # reload best model
        net.load_state_dict(torch.load(save_path, map_location=device))
        best_metrics = val(net, val_loader_fold, loss_fn, device, print_cm=True)

        print(f"\n[Fold {fold+1} BEST RESULTS]")
        for k, v in best_metrics.items():
            print(f"{k}: {v:.4f}")

        fold_results.append(best_metrics)

        # wandb artifact
        artifact = wandb.Artifact(f"best_model_fold_{fold+1}", type="model")
        artifact.add_file(save_path)
        wandb.log_artifact(artifact)

        wandb.finish()
        torch.cuda.empty_cache()

    # FINAL
    print("\n========== FINAL K-FOLD RESULTS ==========")

    keys = fold_results[0].keys()

    for key in keys:
        values = [fr[key] for fr in fold_results]
        print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    print("==========================================")


if __name__ == "__main__":
    main()