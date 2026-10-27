#!/usr/bin/env python3
"""
analisar_se.py

Mostra o que o bloco Squeeze-and-Excitation (SE) do extrator faz: para cada
ataque, o peso (0 a 1) que ele dá a cada sub-banda (LL, LH, HL) na hora de
ler a marca. Se o SE estiver funcionando, os pesos mudam conforme o ataque.

Para cada ataque também mede a bit accuracy com o SE normal e com o SE
"desligado" (todos os pesos iguais à média que ele dá sem ataque), para ver
quanto a escolha dos pesos por imagem ajuda.

Saídas em --output: pesos_se.csv, pesos_se.png

    python analisar_se.py --num_images 200
"""

import os
import random
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from attacks import AttackLayer
from watermark_embedding import WatermarkEmbedding, watermark_embedding_tensor
from watermark_extraction import WatermarkExtraction

CHECKPOINT = "output/checkpoints/v3_final.pth"
VAL_DIR = "Dataset/celeba_hq/val"

ATAQUES = [
    ("sem ataque", "none", {}),
    ("JPEG 75", "jpeg_test", {"quality": 75}),
    ("JPEG 50", "jpeg_test", {"quality": 50}),
    ("JPEG 30", "jpeg_test", {"quality": 30}),
    ("blur σ 2", "blur", {"sigma": 2.0}),
    ("median 5", "median_blur", {"kernel": 5}),
    ("ruído σ 0,05", "noise", {"std": 0.05}),
    ("salt & pepper 5%", "salt_pepper", {"amount": 0.05}),
    ("dropout 20%", "dropout", {"p": 0.2}),
    ("resize 0,5", "resize", {"scale": 0.5}),
    ("resize 0,3", "resize", {"scale": 0.3}),
]
BANDAS = ["LL", "LH", "HL"]


def find_images(root):
    out = []
    for d, _, fs in os.walk(root):
        out += [os.path.join(d, f) for f in fs if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return sorted(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--val_dir", default=VAL_DIR)
    parser.add_argument("--num_images", type=int, default=200)
    parser.add_argument("--output", default="my_tests/se_pesos")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    os.makedirs(args.output, exist_ok=True)

    ck = torch.load(args.checkpoint, map_location=dev)
    emb = WatermarkEmbedding(ck["watermark_size"], 256, num_bits=ck["num_bits"]).to(dev).eval()
    emb.load_state_dict(ck["embedding_model"]); emb.alpha = ck["alpha"]
    ext = WatermarkExtraction(ck["watermark_size"], 256, use_hh=ck.get("use_hh", False), num_bits=ck["num_bits"]).to(dev).eval()
    ext.load_state_dict(ck["extraction_model"])

    # captura os pesos que o SE calcula (saída da sigmoide, um por sub-banda)
    capturado = {}
    ext.se_in.fc.register_forward_hook(lambda m, i, o: capturado.__setitem__("w", o.detach()))

    # SE "desligado": substitui os pesos calculados por pesos fixos
    fixo = {"w": None}
    def se_fixo(mod, inp, out):
        if fixo["w"] is None:
            return None
        return inp[0] * fixo["w"].view(1, -1, 1, 1)
    ext.se_in.register_forward_hook(se_fixo)

    t = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    paths = find_images(args.val_dir)[:args.num_images]
    attack_layer = AttackLayer().to(dev).eval()

    linhas = []
    fixo_ref = None   # pesos médios do SE sem ataque (calculados no primeiro ataque)
    for nome, atk, kw in ATAQUES:
        pesos, acc_se, acc_fixo = [], [], []
        for i, p in enumerate(paths):
            x = t(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
            m = torch.randint(0, 2, (1, ck["num_bits"]), device=dev).float()
            marcada = torch.round(watermark_embedding_tensor(x, m, emb)[0] * 255) / 255
            random.seed(args.seed + i); torch.manual_seed(args.seed + i)
            atacada = attack_layer(marcada, attack=atk, **kw)[0].clamp(0, 1)

            fixo["w"] = None
            acc_se.append(((ext(atacada) >= 0).float() == m).float().mean().item())
            pesos.append(capturado["w"][0].cpu().numpy())
            if fixo_ref is not None:
                fixo["w"] = fixo_ref
                acc_fixo.append(((ext(atacada) >= 0).float() == m).float().mean().item())
        pesos = np.array(pesos)
        if nome == "sem ataque":
            fixo_ref = torch.tensor(pesos.mean(0), device=dev)
        linha = {"Ataque": nome}
        for b, banda in enumerate(BANDAS):
            linha[f"peso {banda}"] = pesos[:, b].mean()
            linha[f"desvio {banda}"] = pesos[:, b].std()
        linha["bit acc (SE normal) %"] = np.mean(acc_se) * 100
        linha["bit acc (SE fixo) %"] = np.mean(acc_fixo) * 100 if acc_fixo else np.nan
        linhas.append(linha)
        print(f"{nome:18s} " + "  ".join(f"{b} {linha[f'peso {b}']:.3f}" for b in BANDAS)
              + f" | acc SE {linha['bit acc (SE normal) %']:.2f}% | SE fixo {linha['bit acc (SE fixo) %']:.2f}%")

    df = pd.DataFrame(linhas)
    df.to_csv(os.path.join(args.output, "pesos_se.csv"), index=False)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    xs = np.arange(len(df))
    cores = {"LL": "#1D5FA8", "LH": "#B45309", "HL": "#7A8B99"}
    for k, banda in enumerate(BANDAS):
        ax.bar(xs + (k - 1) * 0.27, df[f"peso {banda}"], 0.27, yerr=df[f"desvio {banda}"],
               label=banda, color=cores[banda], capsize=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(df["Ataque"], rotation=30, ha="right")
    ax.set_ylabel("peso do SE (0 a 1)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Pesos que o SE dá a cada sub-banda, por ataque ({len(paths)} imagens)")
    ax.legend(title="sub-banda")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "pesos_se.png"), dpi=150)
    print(f"\nCSV e figura em {args.output}")


if __name__ == "__main__":
    main()
