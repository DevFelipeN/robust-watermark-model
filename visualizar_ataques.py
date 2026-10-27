#!/usr/bin/env python3
"""
visualizar_ataques.py

Marca UMA imagem, aplica todos os ataques do evaluate.py e salva, para cada
ataque:

    <output_dir>/atacadas/<ataque>.png   imagem marcada depois do ataque
    <output_dir>/marca/<ataque>.png      marca isolada depois do ataque (amplificada)
    <output_dir>/grid_ataques.png        tudo junto: uma linha por ataque

"Marca isolada" = ataque(marcada) - ataque(original), com a mesma aleatoriedade
nos dois (mesma seed). Assim sobra só o que a marca virou depois do ataque,
sem o efeito do ataque sobre a imagem em si. Todas as linhas usam a mesma
amplificação, então dá para comparar visualmente se algum ataque reforça a
marca (como acontece com o DiffAE) ou a enfraquece.

No terminal, para cada ataque:
    - Ganho: força da marca depois / antes do ataque (>1 = amplificou)
    - Forma: correlação do padrão de detalhes finos com a marca original
    - Bit Accuracy da extração

Exemplo:
    python visualizar_ataques.py \
        --models_path output/checkpoints/watermark_Y_DCT_DWT_07102026_10:52:58_final.pth \
        --image_path Dataset/celeba_hq/val/female/003251.jpg \
        --image_resolution 256
"""

import os
import random
import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from attacks import AttackLayer
from watermark_embedding import WatermarkEmbedding, watermark_embedding_tensor
from watermark_extraction import WatermarkExtraction


# Mesmos ataques e parâmetros do evaluate.py
ATTACK_CASES = [
    ("none",               "none",        {}),

    ("jpeg_test",          "jpeg_test",   {"quality": 75}),
    ("jpeg_test_forte50",  "jpeg_test",   {"quality": 50}),
    ("jpeg_test_forte30",  "jpeg_test",   {"quality": 30}),

    ("blur",               "blur",        {"sigma": 2.0}),

    ("median_blur",        "median_blur", {"kernel": 3}),
    ("median_blur_forte",  "median_blur", {"kernel": 5}),

    ("noise",              "noise",       {"std": 0.02}),
    ("noise_forte",        "noise",       {"std": 0.05}),

    ("salt_pepper",        "salt_pepper", {"amount": 0.01}),

    ("dropout",            "dropout",     {"p": 0.05}),
    ("dropout_forte10",    "dropout",     {"p": 0.1}),
    ("dropout_forte20",    "dropout",     {"p": 0.2}),

    ("resize",             "resize",      {"scale": 0.5}),
    ("resize_forte",       "resize",      {"scale": 0.3}),
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_uint8(tensor):
    return (
        tensor[0].permute(1, 2, 0).detach().cpu().numpy() * 255
    ).round().clip(0, 255).astype(np.uint8)


def luminance(rgb):
    rgb = rgb.astype(np.float64)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def high_pass(y):
    return y - cv2.GaussianBlur(y, (0, 0), 2)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--models_path", required=True)
    parser.add_argument("--image_path", default="Dataset/celeba_hq/val/female/003251.jpg")
    parser.add_argument("--image_resolution", type=int, default=256)
    parser.add_argument("--output_dir", default="my_tests/ataques")
    parser.add_argument("--amplify", type=float, default=10.0, help="Amplificação visual da marca isolada (igual para todos os ataques).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attacked_dir = os.path.join(args.output_dir, "atacadas")
    mark_dir = os.path.join(args.output_dir, "marca")
    os.makedirs(attacked_dir, exist_ok=True)
    os.makedirs(mark_dir, exist_ok=True)

    # -------------------------
    # Modelos (formato da marca lido do checkpoint)
    # -------------------------
    checkpoint = torch.load(args.models_path, map_location=device)
    num_bits = checkpoint["num_bits"]
    grid_size = checkpoint["watermark_size"]

    embedding_model = WatermarkEmbedding(grid_size, args.image_resolution, num_bits=num_bits).to(device).eval()
    embedding_model.load_state_dict(checkpoint["embedding_model"])
    embedding_model.alpha = checkpoint["alpha"]

    extractor_model = WatermarkExtraction(
        grid_size, args.image_resolution, use_hh=checkpoint.get("use_hh", False), num_bits=num_bits
    ).to(device).eval()
    extractor_model.load_state_dict(checkpoint["extraction_model"])

    attack_layer = AttackLayer().to(device)

    # -------------------------
    # Marcar (e quantizar para 8 bits, como uma imagem salva em PNG)
    # -------------------------
    transform = transforms.Compose([
        transforms.Resize((args.image_resolution, args.image_resolution)),
        transforms.ToTensor(),
    ])

    to_tensor = lambda arr: transforms.ToTensor()(arr).unsqueeze(0).to(device)

    image = transform(Image.open(args.image_path).convert("RGB")).unsqueeze(0).to(device)
    watermark = torch.randint(0, 2, (1, num_bits), device=device).float()

    with torch.no_grad():
        marked, _, _ = watermark_embedding_tensor(image, watermark, embedding_model)

    original_np = to_uint8(image)
    marked_np = to_uint8(marked)

    Image.fromarray(original_np).save(os.path.join(args.output_dir, "original.png"))
    Image.fromarray(marked_np).save(os.path.join(args.output_dir, "marcada.png"))

    original_t = to_tensor(original_np)
    marked_t = to_tensor(marked_np)

    mark_before = luminance(marked_np) - luminance(original_np)
    rms_before = np.sqrt((mark_before ** 2).mean())

    print(f"Imagem: {args.image_path}")
    print(f"Marca: {num_bits} bits | força antes dos ataques (RMS): {rms_before:.2f}\n")
    print(f"{'Ataque':20s} {'Parâmetro':14s} {'Ganho':>7s} {'Forma':>7s} {'Bit Acc':>9s}")

    # -------------------------
    # Ataques
    # -------------------------
    rows = []

    with torch.no_grad():
        for key, attack, params in ATTACK_CASES:

            # mesma aleatoriedade (ruído, máscara do dropout...) nas duas versões
            set_seed(args.seed)
            attacked_marked, _ = attack_layer(marked_t, attack=attack, **params)

            set_seed(args.seed)
            attacked_original, _ = attack_layer(original_t, attack=attack, **params)

            attacked_marked_np = to_uint8(attacked_marked.clamp(0, 1))
            attacked_original_np = to_uint8(attacked_original.clamp(0, 1))

            mark_after = luminance(attacked_marked_np) - luminance(attacked_original_np)
            gain = np.sqrt((mark_after ** 2).mean()) / rms_before
            shape = np.corrcoef(high_pass(mark_before).ravel(), high_pass(mark_after).ravel())[0, 1]

            predicted = (extractor_model(to_tensor(attacked_marked_np)) >= 0).float()
            bit_acc = (predicted == watermark).float().mean().item()

            mark_vis = np.clip(128 + mark_after * args.amplify, 0, 255).astype(np.uint8)
            mark_vis = np.stack([mark_vis] * 3, axis=-1)

            Image.fromarray(attacked_marked_np).save(os.path.join(attacked_dir, key + ".png"))
            Image.fromarray(mark_vis).save(os.path.join(mark_dir, key + ".png"))

            param_text = ", ".join(f"{k}={v}" for k, v in params.items()) or "-"
            print(f"{key:20s} {param_text:14s} {gain:6.2f}x {shape:7.2f} {bit_acc * 100:8.2f}%")

            label = f"{key} {param_text} | ganho {gain:.2f}x | bit acc {bit_acc * 100:.1f}%"
            rows.append((label, attacked_marked_np, mark_vis))

    # -------------------------
    # Grid: uma linha por ataque (imagem atacada | marca isolada)
    # -------------------------
    width = args.image_resolution * 2
    blocks = []

    for label, attacked_np, mark_vis in rows:
        header = np.full((22, width, 3), 255, np.uint8)
        cv2.putText(header, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        blocks.append(header)
        blocks.append(np.concatenate([attacked_np, mark_vis], axis=1))

    grid_path = os.path.join(args.output_dir, "grid_ataques.png")
    Image.fromarray(np.concatenate(blocks, axis=0)).save(grid_path)

    print(f"\nGanho > 1: o ataque reforçou a marca | < 1: enfraqueceu.")
    print(f"Forma: 1 = mesmo padrão da marca original | ~0 = padrão destruído.")
    print(f"\nImagens atacadas: {attacked_dir}")
    print(f"Marca isolada (amplificada {args.amplify:g}x): {mark_dir}")
    print(f"Grid: {grid_path}")


if __name__ == "__main__":
    main()
