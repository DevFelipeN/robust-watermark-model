#!/usr/bin/env python3
"""
gerar_imagens_ataques.py

Marca a imagem de referência com o modelo, aplica os ataques do evaluate.py
e salva, para cada ataque, uma pasta com:

    original.png          imagem original (256x256)
    marcada.png           imagem marcada
    marcada_atacada.png   imagem marcada depois do ataque
    grid.png              original | marcada | marcada + ataque, com PSNR e
                          a marca extraída da imagem atacada

e, na pasta de saída, resumo.csv (um ataque por linha).

Cada ataque usa sempre a mesma seed, então rodar um ataque sozinho
(--attack) gera exatamente a mesma imagem que rodar todos.

Exemplos:
    python gerar_imagens_ataques.py                          # todos os ataques
    python gerar_imagens_ataques.py --attack jpeg_test_forte30
    python gerar_imagens_ataques.py --list                   # lista os ataques
"""

import os
import math
import random
import argparse

import numpy as np
import pandas as pd
import torch

import unicodedata

from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from attacks import AttackLayer
from watermark_embedding import WatermarkEmbedding, watermark_embedding_tensor
from watermark_extraction import WatermarkExtraction

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.path.join(
    HERE, "output_grade", "v3_finetune_cosine", "checkpoints",
    "watermark_Y_DCT_DWT_12102026_08:55:13_final.pth")

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
    ("salt_pepper_forte",  "salt_pepper", {"amount": 0.05}),
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


def to_uint8(x_01):
    """tensor (1, 3, H, W) em [0, 1] -> array (H, W, 3) uint8"""
    return (x_01[0].permute(1, 2, 0).detach().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(255 ** 2 / mse)


FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def load_font(size=12):
    """Fonte com acentos; sem ela, usa a padrão do PIL (e o texto sai sem acento)."""
    for path in FONT_PATHS:
        if os.path.exists(path):
            return ImageFont.truetype(path, size), True
    return ImageFont.load_default(), False


def ascii_text(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def params_text(params):
    return ", ".join(f"{k}={v}" for k, v in params.items()) or "-"


def make_grid(panels, footer):
    """panels: lista de (título, array HxWx3). Uma faixa de texto em cima e outra embaixo."""
    h, w = panels[0][1].shape[:2]
    gap, top, bottom = 8, 34, 26
    grid = Image.new("RGB", (len(panels) * w + (len(panels) - 1) * gap, top + h + bottom), "white")
    draw = ImageDraw.Draw(grid)
    font, has_accents = load_font()
    fix = (lambda t: t) if has_accents else ascii_text
    for i, (title, img) in enumerate(panels):
        x = i * (w + gap)
        grid.paste(Image.fromarray(img), (x, top))
        for j, line in enumerate(title.split("\n")):
            draw.text((x + 4, 3 + 14 * j), fix(line), fill="black", font=font)
    draw.text((4, top + h + 7), fix(footer), fill="black", font=font)
    return grid


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models_path", default=DEFAULT_CHECKPOINT, help="Padrão: o modelo atual.")
    parser.add_argument("--image_path", default=os.path.join(HERE, "Dataset/celeba_hq/val/female/003251.jpg"))
    parser.add_argument("--attack", default="all",
                        help="Nome de um ataque (veja --list) ou 'all' para todos (padrão).")
    parser.add_argument("--list", action="store_true", help="Lista os ataques disponíveis e sai.")
    parser.add_argument("--alpha", type=float, default=None, help="Padrão: o alpha do checkpoint.")
    parser.add_argument("--image_resolution", type=int, default=256)
    parser.add_argument("--output_dir", default=None, help="Padrão: my_tests/ataques_<nome da imagem>")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.list:
        for key, _, params in ATTACK_CASES:
            print(f"{key:20s} {params_text(params)}")
        return

    cases = ATTACK_CASES if args.attack == "all" else [c for c in ATTACK_CASES if c[0] == args.attack]
    if not cases:
        raise SystemExit(f"Ataque desconhecido: {args.attack}. Use --list para ver os nomes.")

    stem = os.path.splitext(os.path.basename(args.image_path))[0]
    output_dir = args.output_dir or os.path.join(HERE, "my_tests", f"ataques_{stem}")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Modelo
    # -------------------------
    ck = torch.load(args.models_path, map_location=device)
    num_bits, grid_size = ck["num_bits"], ck["watermark_size"]

    embedding = WatermarkEmbedding(grid_size, args.image_resolution, num_bits=num_bits).to(device).eval()
    embedding.load_state_dict(ck["embedding_model"])
    embedding.alpha = args.alpha if args.alpha is not None else ck["alpha"]

    extractor = WatermarkExtraction(grid_size, args.image_resolution,
                                    use_hh=ck.get("use_hh", False), num_bits=num_bits).to(device).eval()
    extractor.load_state_dict(ck["extraction_model"])

    attack_layer = AttackLayer().to(device).eval()

    # -------------------------
    # Imagem e marca (mensagem fixa pela seed)
    # -------------------------
    tf = transforms.Compose([transforms.Resize((args.image_resolution, args.image_resolution)), transforms.ToTensor()])
    original = tf(Image.open(args.image_path).convert("RGB")).unsqueeze(0).to(device)
    original = torch.round(original * 255) / 255

    generator = torch.Generator().manual_seed(args.seed)
    message = torch.randint(0, 2, (1, num_bits), generator=generator).float().to(device)

    with torch.no_grad():
        marked, _, _ = watermark_embedding_tensor(original, message, embedding)
    marked = torch.round(marked.clamp(0, 1) * 255) / 255

    original_np, marked_np = to_uint8(original), to_uint8(marked)
    message_txt = "".join(str(int(b)) for b in message[0].tolist())

    print(f"Modelo: {args.models_path} (alpha {embedding.alpha})")
    print(f"Imagem: {args.image_path} | marca: {message_txt}")
    print(f"PSNR original x marcada: {psnr(original_np, marked_np):.2f} dB\n")

    # -------------------------
    # Ataques
    # -------------------------
    rows = []
    for index, (key, attack, params) in enumerate(ATTACK_CASES):
        if (key, attack, params) not in cases:
            continue

        set_seed(args.seed + index)          # mesma seed para o ataque, rodando um ou todos
        with torch.no_grad():
            attacked, _ = attack_layer(marked, attack=attack, **params)
            attacked = attacked.clamp(0, 1)
            read = (extractor(attacked) >= 0).float()

        attacked_np = to_uint8(attacked)
        errors = int((read != message).sum().item())
        bit_acc = 100 * (1 - errors / num_bits)
        psnr_attacked = psnr(original_np, attacked_np)

        attack_dir = os.path.join(output_dir, key)
        os.makedirs(attack_dir, exist_ok=True)
        Image.fromarray(original_np).save(os.path.join(attack_dir, "original.png"))
        Image.fromarray(marked_np).save(os.path.join(attack_dir, "marcada.png"))
        Image.fromarray(attacked_np).save(os.path.join(attack_dir, "marcada_atacada.png"))

        read_txt = "".join(str(int(b)) for b in read[0].tolist())
        footer = (f"Marca extraída da atacada: {bit_acc:.1f}% dos bits ({errors} errado(s) de {num_bits})"
                  + (" - marca exata" if errors == 0 else ""))
        grid = make_grid([
            ("Original", original_np),
            (f"Marcada\nPSNR {psnr(original_np, marked_np):.2f} dB", marked_np),
            (f"Marcada + {key}\n{params_text(params)} | PSNR {psnr_attacked:.2f} dB", attacked_np),
        ], footer)
        grid.save(os.path.join(attack_dir, "grid.png"))

        rows.append({"ataque": key, "parametros": params_text(params), "psnr_original_x_atacada": round(psnr_attacked, 2),
                     "bits_errados": errors, "bit_accuracy": round(bit_acc, 2), "marca_exata": int(errors == 0),
                     "marca_extraida": read_txt})
        print(f"{key:20s} {params_text(params):14s} PSNR {psnr_attacked:6.2f} dB | "
              f"bits certos {bit_acc:6.2f}% ({errors} errado(s))")

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(output_dir, "resumo.csv" if args.attack == "all" else f"resumo_{args.attack}.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nImagens em: {output_dir}/<ataque>/\nResumo: {summary_path}")


if __name__ == "__main__":
    main()
