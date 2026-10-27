#!/usr/bin/env python3
"""
avaliar_ablacao.py

Compara os checkpoints da ablação dos pesos das sub-bandas (rodar_ablacao.sh)
de forma justa: pesos diferentes mudam a qualidade da imagem marcada, então
cada modelo é avaliado com o MESMO PSNR (o alpha é ajustado na hora de
marcar, por bisseção, até o PSNR médio bater com --target_psnr) e também no
alpha do treino (4).

Ataques e parâmetros iguais aos do evaluate.py; imagens metade femininas e
metade masculinas.

Saída: output_subbandas/ablacao_resultados.csv e uma tabela no terminal.

Exemplo (de dentro do Robust-Hybrid-Watermarking-cnn-256):
    python output_subbandas/avaliar_ablacao.py --num_images 500
"""

import os
import sys
import glob
import math
import random
import argparse

import lpips
import numpy as np
import pandas as pd
import torch

from PIL import Image
from torchvision import transforms
from torchmetrics.functional.image import structural_similarity_index_measure
from DISTS_pytorch import DISTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from attacks import AttackLayer  # noqa: E402
from watermark_embedding import WatermarkEmbedding, watermark_embedding_tensor  # noqa: E402
from watermark_extraction import WatermarkExtraction  # noqa: E402

ATTACK_CASES = [
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
    ("dropout_forte50",    "dropout",     {"p": 0.5}),
    ("resize",             "resize",      {"scale": 0.5}),
    ("resize_forte",       "resize",      {"scale": 0.3}),
]

# ataques em que os modelos ainda falham: a média deles resume a robustez
HARD = ["jpeg_test_forte50", "jpeg_test_forte30", "median_blur_forte",
        "dropout_forte20", "dropout_forte50", "resize", "resize_forte"]


def limit_by_gender(paths, limit):
    female = [p for p in paths if os.path.basename(os.path.dirname(p)) == "female"]
    male = [p for p in paths if os.path.basename(os.path.dirname(p)) == "male"]
    n_male = limit // 2
    return sorted(female[:limit - n_male] + male[:n_male])


def quantize(x):
    return torch.round(x.clamp(0, 1) * 255) / 255


def psnr_u8(a, b):
    mse = torch.mean((torch.round(a * 255) - torch.round(b * 255)) ** 2).item()
    return 10 * math.log10(255 ** 2 / max(mse, 1e-10))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--celeba_val_dir", default=os.path.join(ROOT, "Dataset/celeba_hq/val"))
    parser.add_argument("--num_images", type=int, default=500)
    parser.add_argument("--target_psnr", type=float, default=47.2, help="PSNR do modelo v2 com alpha 4.")
    parser.add_argument("--native_alpha", type=float, default=None, help="Padrão: o alpha salvo em cada checkpoint.")
    parser.add_argument("--dir", default=os.path.join(ROOT, "output_subbandas"), help="Pasta com <config>/checkpoints/*_final.pth.")
    parser.add_argument("--models", default=None, help="Avaliar só estas configurações (nomes das subpastas, separados por vírgula).")
    parser.add_argument("--no_v2_base", action="store_true", help="Não incluir o checkpoint v2 como referência.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Padrão: <dir>/ablacao_resultados.csv")
    args = parser.parse_args()
    if args.output is None:
        args.output = os.path.join(args.dir, "ablacao_resultados.csv")

    device = "cuda"
    checkpoints = {}
    v2_base = os.path.join(ROOT, "output/checkpoints/watermark_Y_DCT_DWT_09102026_19:41:33_final.pth")
    if not args.no_v2_base and os.path.exists(v2_base):
        checkpoints["v2_base"] = v2_base
    wanted = set(args.models.split(",")) if args.models else None
    for path in sorted(glob.glob(os.path.join(args.dir, "*", "checkpoints", "*_final.pth"))):
        name = path.split(os.sep)[-3]
        if wanted is None or name in wanted:
            checkpoints[name] = path
    if not checkpoints:
        raise ValueError("Nenhum checkpoint encontrado.")

    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    paths = limit_by_gender(sorted(glob.glob(os.path.join(args.celeba_val_dir, "*", "*.jpg"))), args.num_images)
    images = [quantize(tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)) for p in paths]
    calib = images[:: max(1, len(images) // 100)][:100]      # 100 imagens para calibrar o alpha

    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    ds = DISTS().to(device).eval()
    attack_layer = AttackLayer().to(device).eval()

    torch.manual_seed(args.seed)
    rows = []

    for name, path in checkpoints.items():
        ck = torch.load(path, map_location=device)
        nb, grid = ck["num_bits"], ck["watermark_size"]
        emb = WatermarkEmbedding(grid, 256, num_bits=nb).to(device).eval()
        emb.load_state_dict(ck["embedding_model"])
        ext = WatermarkExtraction(grid, 256, use_hh=ck.get("use_hh", False), num_bits=nb).to(device).eval()
        ext.load_state_dict(ck["extraction_model"])
        weights = emb.get_subband_weights().view(4).tolist()

        g = torch.Generator().manual_seed(args.seed)
        msg = torch.randint(0, 2, (1, nb), generator=g).float().to(device)

        def mark(x, alpha):
            emb.alpha = alpha
            with torch.no_grad():
                w, _, _ = watermark_embedding_tensor(x, msg, emb)
            return quantize(w)

        def mean_psnr(alpha):
            return np.mean([psnr_u8(x, mark(x, alpha)) for x in calib])

        # bisseção do alpha que dá o PSNR alvo (PSNR cai quando o alpha sobe)
        lo, hi = 0.5, 20.0
        for _ in range(18):
            mid = (lo + hi) / 2
            if mean_psnr(mid) > args.target_psnr:
                lo = mid
            else:
                hi = mid
        matched_alpha = (lo + hi) / 2

        native = args.native_alpha if args.native_alpha is not None else float(ck["alpha"])
        for mode, alpha in (("mesmo_PSNR", matched_alpha), ("alpha_nativo", native)):
            row = {"modelo": name, "modo": mode, "alpha": round(alpha, 3),
                   "w_LL": weights[0], "w_LH": weights[1], "w_HL": weights[2], "w_HH": weights[3]}
            q, res = [], {k: [] for k, _, _ in ATTACK_CASES}
            folded = torch.zeros(16, 16, device=device)   # |resíduo em Y| por posição na célula de 16x16 px
            for i, x in enumerate(images):
                w = mark(x, alpha)
                with torch.no_grad():
                    dy = 255 * (0.299 * (w[0, 0] - x[0, 0]) + 0.587 * (w[0, 1] - x[0, 1]) + 0.114 * (w[0, 2] - x[0, 2]))
                    folded += dy.abs().reshape(16, 16, 16, 16).mean(dim=(0, 2))
                    q.append((psnr_u8(x, w), structural_similarity_index_measure(x, w).item(),
                              lp(x * 2 - 1, w * 2 - 1).item(), ds(x, w).item()))
                    for key, attack, params in ATTACK_CASES:
                        random.seed(args.seed + i); np.random.seed(args.seed + i); torch.manual_seed(args.seed + i)
                        y, _ = attack_layer(w, attack=attack, **params)
                        bits = ((ext(y.clamp(0, 1)) >= 0).float() == msg).float()
                        res[key].append((bits.mean().item(), float(bits.min().item() == 1)))
            q = np.mean(q, axis=0)
            row.update({"PSNR": q[0], "SSIM": q[1], "LPIPS": q[2], "DISTS": q[3]})
            # padrão em grade: quanto a energia se concentra nas mesmas posições
            # das células de 16x16 px (1 = espalhada por igual)
            row["grade_pico_media"] = (folded.max() / folded.mean()).item()
            row["grade_cv"] = (folded.std() / folded.mean()).item()
            for key in res:
                row[f"exata_{key}"] = 100 * np.mean([v[1] for v in res[key]])
                row[f"bitacc_{key}"] = 100 * np.mean([v[0] for v in res[key]])
            row["exata_media_dificeis"] = np.mean([row[f"exata_{k}"] for k in HARD])
            rows.append(row)
            print(f"{name:13s} {mode:12s} alpha={alpha:5.2f} PSNR={q[0]:.2f} LPIPS={q[2]:.4f} "
                  f"DISTS={q[3]:.4f} grade={row['grade_pico_media']:.2f} | exata média difíceis={row['exata_media_dificeis']:.1f}%", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    show = ["modelo", "alpha", "w_LL", "w_LH", "w_HL", "PSNR", "SSIM", "LPIPS", "DISTS", "grade_pico_media", "exata_media_dificeis"] + \
           [f"exata_{k}" for k in HARD]
    for mode in ("mesmo_PSNR", "alpha_nativo"):
        print(f"\n=== {mode} ({len(images)} imagens) — % de imagens com a marca exata")
        print(df[df.modo == mode][show].round(3).to_string(index=False))
    print(f"\nCSV: {args.output}")


if __name__ == "__main__":
    main()
