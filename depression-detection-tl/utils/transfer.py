# import torch

# def load_pretrained(net, path, device):
#     print(f"[TL] Loading pretrained weights from: {path}")
#     state_dict = torch.load(path, map_location=device)
#     net.load_state_dict(state_dict, strict=False)
#     return net


# def freeze_all(net):
#     for param in net.parameters():
#         param.requires_grad = False


# def unfreeze_classifier(net):
#     for name, param in net.named_parameters():
#         if any(key in name for key in ["fc", "output"]):
#             param.requires_grad = True


# def unfreeze_last_layers(net, keywords):
#     for name, param in net.named_parameters():
#         if any(k in name for k in keywords):
#             param.requires_grad = True


# def freeze_partial_transformer(net, num_freeze=3):          ### added
#     """
#     Freeze first N layers of each transformer encoder
#     """

#     encoders = [
#         net.a_encoder,
#         net.v_encoder,
#         net.av_encoder
#     ]

#     for encoder in encoders:
#         for i, layer in enumerate(encoder.layers):
#             if i < num_freeze:
#                 for param in layer.parameters():
#                     param.requires_grad = False

# # Approach 1---------- unFreeze only fc layer-------------
# # def apply_transfer_learning(net, args):
# #     """
# #     Fully controlled by config.yaml
# #     """

# #     if args.pretrained_path is None:
# #         print("[TL] No pretrained model used")
# #         return net

# #     # Load weights
# #     net = load_pretrained(net, args.pretrained_path, args.device[0])

# #     if args.freeze_backbone:
# #         print("[TL] Freezing backbone...")
# #         freeze_all(net)
# #         unfreeze_classifier(net)

# #     if args.finetune_layers:
# #         print("[TL] Fine-tuning last layers...")
# #         keywords = ["encoder", "gtcn", "transformer"]
# #         unfreeze_last_layers(net, keywords)

# #     return net


# # #  Approach 2 --------------Freeze first N layers of each transformer encoder-------------
# # def apply_transfer_learning(net, args):

# #     if args.pretrained_path is None:
# #         print("[TL] No pretrained model used")
# #         return net

# #     # Load weights
# #     net = load_pretrained(net, args.pretrained_path, args.device[0])

# #     # STEP 1: Freeze everything
# #     if args.freeze_backbone:
# #         print("[TL] Freezing backbone...")
# #         freeze_all(net)

# #         # Always train classifier
# #         unfreeze_classifier(net)

# #     # STEP 2: Partial fine-tuning
# #     if args.finetune_layers:
# #         print("[TL] Fine-tuning partial transformer layers...")

# #         # Unfreeze ALL transformer layers first
# #         for name, param in net.named_parameters():
# #             if "encoder" in name:
# #                 param.requires_grad = True

# #         # Now freeze early layers again
# #         freeze_partial_transformer(net, num_freeze=3)

# #     return net

# # Approach 3-----------------------------
# # =========================
# # PARTIAL MULTIMODAL FINETUNING
# # =========================
# def freeze_unimodal_and_partial_multimodal(net, num_freeze=1):
#     """
#     Freeze:
#     - a_encoder (audio)
#     - v_encoder (visual)

#     Train:
#     - last layers of av_encoder
#     """

#     # Freeze unimodal encoders completely
#     for param in net.a_encoder.parameters():
#         param.requires_grad = False

#     for param in net.v_encoder.parameters():
#         param.requires_grad = False

#     # Multimodal encoder (partial)
#     for i, layer in enumerate(net.av_encoder.layers):
#         if i < num_freeze:
#             # freeze early layers
#             for param in layer.parameters():
#                 param.requires_grad = False
#         else:
#             # train later layers
#             for param in layer.parameters():
#                 param.requires_grad = True


# # =========================
# # APPLY TRANSFER LEARNING
# # =========================
# def apply_transfer_learning(net, args):
#     """
#     Controlled by config.yaml
#     """

#     # No TL
#     if args.pretrained_path is None:
#         print("[TL] No pretrained model used")
#         return net

#     # Load pretrained weights
#     net = load_pretrained(net, args.pretrained_path, args.device[0])

#     # STEP 1: Freeze everything
#     if args.freeze_backbone:
#         print("[TL] Freezing backbone...")
#         freeze_all(net)

#         # Always train classifier
#         unfreeze_classifier(net)

#     # STEP 2: Partial fine-tuning (multimodal only)
#     if args.finetune_layers:
#         print("[TL] Fine-tuning multimodal last layers only...")

#         freeze_unimodal_and_partial_multimodal(net, num_freeze=2)

#         # Ensure classifier stays trainable
#         unfreeze_classifier(net)

#     return net


####-------------------Fresh start-------------------
import torch


# =========================
# LOAD PRETRAINED
# =========================
def load_pretrained(net, path, device):
    print(f"[TL] Loading pretrained weights from: {path}")
    state_dict = torch.load(path, map_location=device)
    net.load_state_dict(state_dict, strict=False)
    return net


# =========================
# FREEZE ALL
# =========================
def freeze_all(net):
    for param in net.parameters():
        param.requires_grad = False


# =========================
# UNFREEZE CLASSIFIER
# =========================
def unfreeze_classifier(net):
    for name, param in net.named_parameters():
        if "fc" in name or "output" in name:
            param.requires_grad = True


# =========================
# APPROACH 2: ALL TRANSFORMERS PARTIAL
# =========================
def freeze_partial_all_transformers(net, num_freeze):
    encoders = [net.a_encoder, net.v_encoder, net.av_encoder]

    for encoder in encoders:
        for i, layer in enumerate(encoder.layers):
            if i < num_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True


# =========================
# APPROACH 3: MULTIMODAL ONLY
# =========================
def freeze_partial_multimodal(net, num_freeze):

    # ❄️ Freeze unimodal encoders completely
    for param in net.a_encoder.parameters():
        param.requires_grad = False

    for param in net.v_encoder.parameters():
        param.requires_grad = False

    #  Partial multimodal
    for i, layer in enumerate(net.av_encoder.layers):
        if i < num_freeze:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True


# =========================
# MAIN FUNCTION
# =========================
def apply_transfer_learning(net, args):

    if args.pretrained_path is None:
        print("[TL] No pretrained model used")
        return net

    # Load weights
    net = load_pretrained(net, args.pretrained_path, args.device[0])

    # STEP 1: Freeze everything
    if args.freeze_backbone:
        print("[TL] Freezing backbone...")
        freeze_all(net)

    # STEP 2: Apply strategy
    if args.tl_mode == "fc":
        print("[TL] Training only classifier")
        unfreeze_classifier(net)

    elif args.tl_mode == "all":
        print(f"[TL] Partial fine-tuning ALL transformers (freeze first {args.num_freeze})")

        # unfreeze transformer layers
        for name, param in net.named_parameters():
            if any(k in name for k in [                 #previously it was..... if "encoder" in name:
                "encoder", 
                "transform", 
                "cross", 
                "norm"]):
                param.requires_grad = True

        freeze_partial_all_transformers(net, args.num_freeze)
        unfreeze_classifier(net)

    elif args.tl_mode == "multimodal":
        print(f"[TL] Partial fine-tuning MULTIMODAL only (freeze first {args.num_freeze})")

        freeze_partial_multimodal(net, args.num_freeze)
        unfreeze_classifier(net)

    else:
        raise ValueError(f"Unknown tl_mode: {args.tl_mode}")

    return net
