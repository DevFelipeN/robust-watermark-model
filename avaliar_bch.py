#!/usr/bin/env python3
"""
avaliar_bch.py

Antes x depois do código corretor de erros BCH(30, 15) (bch_codec.py), sem
retreinar: o modelo continua escondendo 30 bits, mas agora eles são
15 bits de mensagem + 15 de paridade, e até 3 bits errados são corrigidos.

Cada imagem recebe uma mensagem de 15 bits diferente (sorteada com seed fixa).
Ataques e parâmetros iguais aos do evaluate.py.

Para cada ataque:
    acc_bits          % de bits certos (dos 30 escondidos)
    exata_sem_ECC     % de imagens com os 30 bits certos (como no evaluate.py)
    exata_com_ECC     % de imagens com a mensagem de 15 bits recuperada
    acc_bits_com_ECC  % dos 15 bits de mensagem certos depois do BCH (quando o
                      BCH não consegue corrigir, usa os 15 bits como foram lidos)
    falha_detectada   % em que o BCH avisa que não conseguiu corrigir
    erro_silencioso   % em que o BCH devolve uma mensagem ERRADA sem avisar

Exemplo (opção 3 + geométrico, alpha do mesmo PSNR dos outros modelos):
    python avaliar_bch.py --alpha 5.312
"""

import os
import glob
import random
import argparse

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from attacks import AttackLayer
from bch_codec import BCH3015
from watermark_embedding import WatermarkEmbedding, watermark_embedding_tensor
from watermark_extraction import WatermarkExtraction

HERE = os.path.dirname(os.path.abspath(__file__))

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


def find_images(root):
    return sorted(p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                  if p.lower().endswith((".jpg", ".jpeg", ".png")))


def limit_by_gender(paths, limit):
    female = [p for p in paths if os.path.basename(os.path.dirname(p)) == "female"]
    male = [p for p in paths if os.path.basename(os.path.dirname(p)) == "male"]
    n_male = limit // 2
    return sorted(female[:limit - n_male] + male[:n_male])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_path", default=None,
                        help="Padrão: output/checkpoints/v3_final.pth")
    parser.add_argument("--alpha", type=float, default=None, help="Padrão: o alpha do checkpoint.")
    parser.add_argument("--celeba_val_dir", default=os.path.join(HERE, "Dataset/celeba_hq/val"))
    parser.add_argument("--limit", type=int, default=None, help="N imagens, metade de cada gênero (padrão: todas).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=os.path.join(HERE, "results", "bch_2000.csv"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_path = args.models_path or os.path.join(HERE, "output", "checkpoints", "v3_final.pth")

    ck = torch.load(models_path, map_location=device)
    nb, grid = ck["num_bits"], ck["watermark_size"]
    if nb != BCH3015.CODE_BITS:
        raise ValueError(f"O BCH(30, 15) precisa de um modelo de 30 bits (este tem {nb}).")

    emb = WatermarkEmbedding(grid, 256, num_bits=nb).to(device).eval()
    emb.load_state_dict(ck["embedding_model"])
    emb.alpha = args.alpha if args.alpha is not None else ck["alpha"]
    ext = WatermarkExtraction(grid, 256, use_hh=ck.get("use_hh", False), num_bits=nb).to(device).eval()
    ext.load_state_dict(ck["extraction_model"])
    print(f"Modelo: {models_path} | alpha {emb.alpha}")

    codec = BCH3015()
    attack_layer = AttackLayer().to(device).eval()
    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    paths = find_images(args.celeba_val_dir)
    if args.limit:
        paths = limit_by_gender(paths, args.limit)
    print(f"Imagens: {len(paths)}")

    rng = np.random.default_rng(args.seed)
    rows = []

    for i, path in enumerate(tqdm(paths)):
        msg15 = rng.integers(0, 2, codec.DATA_BITS).astype(np.uint8)   # mensagem desta imagem
        code30 = codec.encode(msg15)
        bits = torch.tensor(code30, dtype=torch.float32, device=device).unsqueeze(0)

        x = torch.round(tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device) * 255) / 255
        with torch.no_grad():
            w, _, _ = watermark_embedding_tensor(x, bits, emb)
        w = torch.round(w.clamp(0, 1) * 255) / 255

        for key, attack, params in ATTACK_CASES:
            random.seed(args.seed + i); np.random.seed(args.seed + i); torch.manual_seed(args.seed + i)
            with torch.no_grad():
                y, _ = attack_layer(w, attack=attack, **params)
                read = (ext(y.clamp(0, 1)) >= 0).cpu().numpy()[0].astype(np.uint8)

            errors = int((read != code30).sum())
            decoded, _ = codec.decode(read)
            if decoded is None:
                outcome = "falha_detectada"
            elif np.array_equal(decoded, msg15):
                outcome = "recuperada"
            else:
                outcome = "erro_silencioso"
            # bits de mensagem certos depois do BCH; sem correção possível, a
            # mensagem é o que foi lido (o código é sistemático: os 15 primeiros
            # bits escondidos são a própria mensagem)
            final15 = read[:codec.DATA_BITS] if decoded is None else decoded
            msg_errors = int((final15 != msg15).sum())
            rows.append({"Image": os.path.basename(path), "Attack": key, "bits_errados": errors,
                         "exata_sem_ECC": int(errors == 0), "resultado_ECC": outcome,
                         "bits_msg_errados_com_ECC": msg_errors})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output.replace(".csv", "_por_imagem.csv"), index=False)

    g = df.groupby("Attack", sort=False)
    summary = pd.DataFrame({
        "acc_bits": 100 * (1 - g["bits_errados"].mean() / codec.CODE_BITS),
        "exata_sem_ECC": 100 * g["exata_sem_ECC"].mean(),
        "exata_com_ECC": 100 * g["resultado_ECC"].apply(lambda s: (s == "recuperada").mean()),
        "acc_bits_com_ECC": 100 * (1 - g["bits_msg_errados_com_ECC"].mean() / codec.DATA_BITS),
        "falha_detectada": 100 * g["resultado_ECC"].apply(lambda s: (s == "falha_detectada").mean()),
        "erro_silencioso": 100 * g["resultado_ECC"].apply(lambda s: (s == "erro_silencioso").mean()),
    })
    summary["ganho"] = summary["exata_com_ECC"] - summary["exata_sem_ECC"]
    summary.to_csv(args.output)

    print(f"\nBCH(30, 15): 15 bits de mensagem, corrige até 3 erros — {len(paths)} imagens, alpha {emb.alpha}")
    print(summary.round(2).to_string())
    print(f"\nCSV: {args.output}")


if __name__ == "__main__":
    main()
