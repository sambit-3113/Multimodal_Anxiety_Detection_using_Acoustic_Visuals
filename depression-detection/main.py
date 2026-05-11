import argparse
import yaml

import wandb
import torch
from tqdm import tqdm

from models import TMeanNet, DepressionDetector, TAMFN  #, ChunkCrossAttentionNet
from models.chunk_cross_improved import ChunkCrossAttentionNet
from models.chunk_transformer import ChunkTransformerNet
from datasets import get_dvlog_dataloader
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

    # Reproducibility settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(
        description="Train and test a model on the DVLOG dataset."
    )
    # arguments whose default values are in config.yaml
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--train_gender", type=str)
    parser.add_argument("--test_gender", type=str)
    parser.add_argument(
        "-m", "--model", type=str,
        choices=["TMeanNet", "DepressionDetector", "TAMFN", "ChunkCrossAttentionNet"]
    )
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-bs", "--batch_size", type=int)
    parser.add_argument("-lr", "--learning_rate", type=float)
    parser.add_argument(
        "-sch", "--lr_scheduler", type=str,
        choices=["cos", "None",]
    )
    parser.add_argument("-d", "--device", type=str, nargs="*")

    # ⭐ Added seed argument
    parser.add_argument("--seed", type=int)

    parser.set_defaults(**config)
    args = parser.parse_args()

    return args


def train_epoch(
    net, train_loader, loss_fn, optimizer, lr_scheduler, device, 
    current_epoch, total_epochs
):
    """One training epoch.
    """
    net.train()
    sample_count = 0
    running_loss = 0.
    correct_count = 0
    TP, FP, TN, FN = 0, 0, 0, 0

    with tqdm(
        train_loader, desc=f"Training epoch {current_epoch}/{total_epochs}",
        leave=False, unit="batch"
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
            # binary classification with only one output neuron
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
    print(f"Actual Depressed       : {TP + FN}")
    print(f"Actual Non-Depressed   : {TN + FP}")
    print(f"Predicted Depressed    : {TP + FP}")
    print(f"Predicted Non-Depressed: {TN + FN}")
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    print("===============================================")    

    return {
        "loss": running_loss / sample_count,
        "acc": correct_count / sample_count,
    }


def val(
    net, val_loader, loss_fn, device, print_cm = False
):
    """Test the model on the validation / test set.
    """
    net.eval()
    sample_count = 0
    running_loss = 0.
    TP, FP, TN, FN = 0, 0, 0, 0

    with torch.no_grad():
        with tqdm(
            val_loader, desc="Validating", leave=False, unit="batch"
        ) as pbar:
            for x, y in pbar:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                y_pred = net(x)
                loss = loss_fn(y_pred, y.to(torch.float32))

                sample_count += x.shape[0]
                running_loss += loss.item() * x.shape[0]
                # binary classification with only one output neuron
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
                accuracy = (
                    (TP + TN) / sample_count
                    if sample_count > 0 else 0.0
                )

                pbar.set_postfix({
                    "loss": l, "acc": accuracy,
                    "precision": precision, "recall": recall, "f1": f1_score,
                })

    l = running_loss / sample_count
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = (
        2 * (precision * recall) / (precision + recall) 
        if (precision + recall) > 0 else 0.0
    )
    accuracy = (
        (TP + TN) / sample_count
        if sample_count > 0 else 0.0
    )

    # ⭐ Balanced Accuracy
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    balanced_acc = (recall + specificity) / 2

    # ⭐ MCC
    denominator = ((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)) ** 0.5
    mcc = ((TP * TN - FP * FN) / denominator) if denominator > 0 else 0.0

    # print(f"Predicted anxious: {TP + FP}")
    # print(f"Predicted non-anxious: {TN + FN}")

    # ⭐ Weighted Precision / Recall / F1
    # Positive class (anxious)
    precision_pos = precision
    recall_pos = recall
    f1_pos = f1_score

    # Negative class
    precision_neg = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    recall_neg = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1_neg = (
        2 * precision_neg * recall_neg / (precision_neg + recall_neg)
        if (precision_neg + recall_neg) > 0 else 0.0
    )

    # Class supports
    N_pos = TP + FN
    N_neg = TN + FP
    N_total = N_pos + N_neg

    weighted_precision = (
        (N_pos * precision_pos + N_neg * precision_neg) / N_total
        if N_total > 0 else 0.0
    )

    weighted_recall = (
        (N_pos * recall_pos + N_neg * recall_neg) / N_total
        if N_total > 0 else 0.0
    )

    weighted_f1 = (
        (N_pos * f1_pos + N_neg * f1_neg) / N_total
        if N_total > 0 else 0.0
    )
    
    if print_cm:
        print("\n========== CONFUSION MATRIX ==========")
        print(f"Actual Positive       : {TP + FN}")
        print(f"Actual Negative       : {TN + FP}")
        print(f"Predicted Positive    : {TP + FP}")
        print(f"Predicted Negative    : {TN + FN}")
        print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
        print("======================================")

    return {
        "loss": l, "acc": accuracy,
        "precision": precision, "recall": recall, "f1": f1_score,
        "balanced_acc": balanced_acc,                                       #### i added
        "mcc": mcc,                                                         #### i added
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }


def main():
    args = parse_args()

    # ⭐ SET RANDOM SEED HERE
    set_seed(args.seed)


    # initialize wandb
    wandb_run_name = f"{args.model}-{args.train_gender}-{args.test_gender}"
    wandb.init(
        project="dvlog", entity="sambitsahoo-k-iiser-bhopal", config=args, name=wandb_run_name,
    )
    args = wandb.config
    print(args)

    # construct the model
    if args.model == "TMeanNet":
        net = TMeanNet(hidden_sizes=[512, 512, 512])
    elif args.model == "DepressionDetector":
        net = DepressionDetector(d=256, l=6, t_downsample=4)
    elif args.model == "TAMFN":
        net = TAMFN(d=256, l=6, t_downsample=4)
    # elif args.model == "ChunkCrossAttentionNet":
    #     net = ChunkCrossAttentionNet(
    #         d_model=128,
    #         chunk_size=20,
    #         num_heads=4,
    #         hidden_dim=128,
    #         dropout=0.5,
    # )

     # ===== NEW MODELS =====
    elif args.model == "ChunkCrossAttentionNet":
        """
        Improved chunk-wise cross attention with bidirectional GRU
        
        Hyperparameters:
        - d_model: 256 (embedding dimension)
        - chunk_size: 20 (frames per chunk at 1 FPS)
        - num_heads: 8 (attention heads)
        - dropout: 0.5
        - num_gru_layers: 2 (bidirectional LSTM)
        """
        net = ChunkCrossAttentionNet(
            d_model=256,
            chunk_size=20,
            num_heads=8,
            dropout=0.5,
            num_gru_layers=2
        )
    
    elif args.model == "ChunkTransformerNet":
        """
        Advanced chunk-wise transformer model
        
        Hyperparameters:
        - d_model: 256 (embedding dimension)
        - chunk_size: 20 (frames per chunk)
        - num_heads: 8 (attention heads)
        - num_encoder_layers: 3 (transformer encoder layers)
        - dropout: 0.5
        """
        net = ChunkTransformerNet(
            d_model=256,
            chunk_size=20,
            num_heads=8,
            num_encoder_layers=3,
            dropout=0.5
        )
    
    else:
        raise ValueError(f"Unknown model: {args.model}")
    

    net = net.to(args.device[0])
    if len(args.device) > 1:
        net = torch.nn.DataParallel(net, device_ids=args.device)

    # prepare the data
    train_loader = get_dvlog_dataloader(
        args.data_dir, "train", args.batch_size, args.train_gender
    )
    val_loader = get_dvlog_dataloader(
        args.data_dir, "valid", args.batch_size, args.test_gender
    )
    test_loader = get_dvlog_dataloader(
        args.data_dir, "test", args.batch_size, args.test_gender
    )

    # set other training components
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    if args.lr_scheduler == "cos":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs // 5, eta_min=args.learning_rate / 20 
        )
    else:
        lr_scheduler = None

    best_val_acc = -1.0
    for epoch in range(args.epochs):
        train_results = train_epoch(
            net, train_loader, loss_fn, optimizer, lr_scheduler, 
            args.device[0], epoch, args.epochs
        )
        val_results = val(net, val_loader, loss_fn, args.device[0])

        # print(
        #     f"Epoch [{epoch+1}/{args.epochs}] "
        #     f"Train Loss: {train_results['loss']:.4f}, "
        #     f"Val Loss: {val_results['loss']:.4f}"
        # )
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
            torch.save(net.state_dict(), f"{wandb.run.dir}/best_model.pt")

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

    # upload the best model to wandb website
    artifact = wandb.Artifact("best_model", type="model")
    artifact.add_file(f"{wandb.run.dir}/best_model.pt")
    wandb.log_artifact(artifact)

    # load the best model for testing
    net.load_state_dict(
        torch.load(f"{wandb.run.dir}/best_model.pt", map_location=args.device[0])
    )
    test_results = val(net, test_loader, loss_fn, args.device[0], print_cm=True)
    print("Test results:")
    print(test_results)
    # print("\n" + "="*60)
    # print("TEST RESULTS:")
    # print("="*60)
    # for key, value in test_results.items():
    #     print(f"{key}: {value:.4f}")
    # print("="*60 + "\n")

    wandb.run.summary["acc/test_acc"] = test_results["acc"]
    wandb.run.summary["loss/test_loss"] = test_results["loss"]
    wandb.run.summary["precision/test_precision"] = test_results["precision"]
    wandb.run.summary["recall/test_recall"] = test_results["recall"]
    wandb.run.summary["f1/test_f1"] = test_results["f1"]
    wandb.run.summary["balanced_acc/test_balanced_acc"] = test_results["balanced_acc"]
    wandb.run.summary["mcc/test_mcc"] = test_results["mcc"]

    wandb.finish()


if __name__ == '__main__':
    main()