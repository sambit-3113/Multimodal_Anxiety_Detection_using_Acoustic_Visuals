import argparse
import yaml

import wandb
import torch
from tqdm import tqdm

from models import TMeanNet, DepressionDetector, TAMFN

from datasets import get_dvlog_window_dataloader

import random
import numpy as np


CONFIG_PATH = "./config.yaml"


# =========================
# RANDOM SEED FUNCTION
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(
        description="Train and test a model on the DVLOG dataset."
    )

    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--train_gender", type=str)
    parser.add_argument("--test_gender", type=str)

    parser.add_argument(
        "-m", "--model", type=str,
        choices=[
            "TMeanNet",
            "DepressionDetector",
            "TAMFN",
            "ChunkCrossAttentionNet",
            "ChunkTransformerNet",
            "SimpleANN"
        ]
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

    # NEW WINDOW PARAMETERS
    parser.add_argument("--window_size", type=int)
    parser.add_argument("--overlap", type=float)

    parser.set_defaults(**config)

    args = parser.parse_args()

    return args


# =========================
# TRAINING FUNCTION
# =========================
def train_epoch(
    net, train_loader, loss_fn, optimizer, lr_scheduler, device,
    current_epoch, total_epochs
):

    net.train()

    sample_count = 0
    running_loss = 0.
    correct_count = 0

    TP, FP, TN, FN = 0, 0, 0, 0

    with tqdm(
        train_loader,
        desc=f"Training epoch {current_epoch}/{total_epochs}",
        leave=False,
        unit="batch"
    ) as pbar:

        for x, y in pbar:

            x, y = x.to(device), y.to(device).unsqueeze(1)

            y_pred = net(x)

            loss = loss_fn(y_pred, y.to(torch.float32))

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            sample_count += x.shape[0]

            running_loss += loss.item() * x.shape[0]

            pred = (y_pred > 0).int()

            correct_count += (pred == y).sum().item()

            TP += torch.sum((pred == 1) & (y == 1)).item()
            FP += torch.sum((pred == 1) & (y == 0)).item()
            TN += torch.sum((pred == 0) & (y == 0)).item()
            FN += torch.sum((pred == 0) & (y == 1)).item()

            pbar.set_postfix({
                "loss": running_loss / sample_count,
                "acc": correct_count / sample_count,
            })

    if lr_scheduler is not None:
        lr_scheduler.step()

    print("\n========== TRAINING CONFUSION MATRIX ==========")
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    print("===============================================")

    return {
        "loss": running_loss / sample_count,
        "acc": correct_count / sample_count,
    }


# =========================
# VALIDATION FUNCTION
# =========================
def val(net, val_loader, loss_fn, device):

    net.eval()

    sample_count = 0
    running_loss = 0.

    TP, FP, TN, FN = 0, 0, 0, 0

    with torch.no_grad():

        with tqdm(val_loader, desc="Validating", leave=False) as pbar:

            for x, y in pbar:

                x, y = x.to(device), y.to(device).unsqueeze(1)

                y_pred = net(x)

                loss = loss_fn(y_pred, y.to(torch.float32))

                sample_count += x.shape[0]

                running_loss += loss.item() * x.shape[0]

                pred = (y_pred > 0).int()

                TP += torch.sum((pred == 1) & (y == 1)).item()
                FP += torch.sum((pred == 1) & (y == 0)).item()
                TN += torch.sum((pred == 0) & (y == 0)).item()
                FN += torch.sum((pred == 0) & (y == 1)).item()

    l = running_loss / sample_count

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    accuracy = (TP + TN) / sample_count

    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    balanced_acc = (recall + specificity) / 2

    denominator = ((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)) ** 0.5

    mcc = ((TP * TN - FP * FN) / denominator) if denominator > 0 else 0.0

    precision_neg = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    recall_neg = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    f1_neg = (
        2 * precision_neg * recall_neg /
        (precision_neg + recall_neg)
        if (precision_neg + recall_neg) > 0 else 0.0
    )

    N_pos = TP + FN
    N_neg = TN + FP
    N_total = N_pos + N_neg

    weighted_precision = (
        (N_pos * precision + N_neg * precision_neg) / N_total
        if N_total > 0 else 0.0
    )

    weighted_recall = (
        (N_pos * recall + N_neg * recall_neg) / N_total
        if N_total > 0 else 0.0
    )

    weighted_f1 = (
        (N_pos * f1_score + N_neg * f1_neg) / N_total
        if N_total > 0 else 0.0
    )

    return {
        "loss": l,
        "acc": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "balanced_acc": balanced_acc,
        "mcc": mcc,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }


# =========================
# MAIN
# =========================
def main():

    args = parse_args()

    set_seed(args.seed)

    wandb_run_name = f"{args.model}-{args.train_gender}-{args.test_gender}"

    wandb.init(
        project="dvlog",
        entity="sambitsahoo-k-iiser-bhopal",
        config=args,
        name=wandb_run_name,
    )

    args = wandb.config

    print(args)

    if args.model == "TMeanNet":
        net = TMeanNet(hidden_sizes=[512, 512, 512])

    elif args.model == "DepressionDetector":
        net = DepressionDetector(d=256, l=6, t_downsample=4)

    elif args.model == "TAMFN":
        net = TAMFN(d=256, l=6, t_downsample=4)

    elif args.model == "SimpleANN":
        from models.ann import SimpleANN
    
        net = SimpleANN(
            input_dim=161,
            hidden_sizes=[256, 128, 64],
            dropout=0.5
        )

    else:
        raise ValueError(f"Unknown model: {args.model}")

    net = net.to(args.device[0])

    # CHANGED DATASET (WINDOW VERSION)
    train_loader = get_dvlog_window_dataloader(
        args.data_dir,
        "train",
        args.batch_size,
        args.train_gender,
        window_size=args.window_size,
        overlap=args.overlap
    )

    val_loader = get_dvlog_window_dataloader(
        args.data_dir,
        "valid",
        args.batch_size,
        args.test_gender,
        window_size=args.window_size,
        overlap=args.overlap
    )

    test_loader = get_dvlog_window_dataloader(
        args.data_dir,
        "test",
        args.batch_size,
        args.test_gender,
        window_size=args.window_size,
        overlap=args.overlap
    )

    def print_label_stats(loader, name="Dataset"):
        labels = [int(y) for _, y in loader.dataset]
        total = len(labels)
        anxious = sum(labels)
        non_anxious = total - anxious
        print(f"\n {name} Label Distribution:")
        print(f"Total samples   : {total}")
        print(f"Anxious (1)     : {anxious}")
        print(f"Non-Anxious (0) : {non_anxious}")
        print(f"Anxious %       : {anxious/total:.2f}")
        print(f"Non-Anxious %   : {non_anxious/total:.2f}")


    print("Train windows:", len(train_loader.dataset))
    print("Validation windows:", len(val_loader.dataset))
    print("Test windows:", len(test_loader.dataset))
    print_label_stats(train_loader, "Train")
    print_label_stats(val_loader, "Validation")
    print_label_stats(test_loader, "Test")



    loss_fn = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    if args.lr_scheduler == "cos":

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs // 5,
            eta_min=args.learning_rate / 20
        )

    else:
        lr_scheduler = None

    best_val_acc = -1.0
    best_model_path = f"{wandb.run.dir}/best_model.pt"

    for epoch in range(args.epochs):

        train_results = train_epoch(
            net,
            train_loader,
            loss_fn,
            optimizer,
            lr_scheduler,
            args.device[0],
            epoch,
            args.epochs
        )

        val_results = val(net, val_loader, loss_fn, args.device[0])

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_results['loss']:.4f}, "
            f"Val Loss: {val_results['loss']:.4f}, "
            f"Val Acc: {val_results['acc']:.4f}, "
            f"Val F1: {val_results['f1']:.4f}"
        )

        val_acc = val_results["acc"]

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), best_model_path)
            print(f" Best model saved with Val Acc: {val_acc:.4f}")

        wandb.log({
            "loss/train_loss": train_results["loss"],
            "acc/train_acc": train_results["acc"],
            "loss/val_loss": val_results["loss"],
            "acc/val_acc": val_results["acc"],
            "precision/val_precision": val_results["precision"],
            "recall/val_recall": val_results["recall"],
            "f1/val_f1": val_results["f1"],
        })

    wandb.run.summary["acc/best_val_acc"] = best_val_acc

    artifact = wandb.Artifact("best_model", type="model")
    artifact.add_file(best_model_path)
    wandb.log_artifact(artifact)

    net.load_state_dict(
        torch.load(best_model_path, map_location=args.device[0])
    )

    test_results = val(net, test_loader, loss_fn, args.device[0])

    print("Test results:")
    print(test_results)

    wandb.finish()


if __name__ == "__main__":
    main()