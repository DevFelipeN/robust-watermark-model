#!/usr/bin/env python3
"""
test_diffae_amostras.py

Teste rápido da marca contra o DiffAE em poucas imagens:

    1. escolhe N imagens de validação (sempre inclui a imagem de referência);
    2. marca cada uma com uma marca aleatória;
    3. manipula as imagens marcadas com o DiffAE (atributo + strength);
    4. extrai a marca antes e depois do DiffAE e mostra a Bit Accuracy;
    5. salva um grid (original | marcada | DiffAE) por imagem.

Saída em <output_dir>/:
    originais/       imagens originais redimensionadas
    marcadas/        imagens marcadas (entrada do DiffAE)
    diffae/          imagens manipuladas pelo DiffAE
    grid_diffae.png  grid comparativo

Exemplo:
    python test_diffae_amostras.py \
        --models_path output/checkpoints/watermark_Y_DCT_DWT_07102026_10:52:58_final.pth \
        --image_resolution 256 \
        --attribute Smiling \
        --strength 0.3
"""

import os
import sys
import glob
import random
import argparse
import subprocess

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from watermark_embedding import WatermarkEmbedding, watermark_embedding_tensor
from watermark_extraction import WatermarkExtraction


DIFFAE_PYTHON = "/home/cnt-computadores/miniconda3/envs/diffae/bin/python"

DIFFAE_DIRS = {
    128: "/home/cnt-computadores/Área de trabalho/FelipeNascimento/models_deepfake/diffae128",
    256: "/home/cnt-computadores/Área de trabalho/FelipeNascimento/models_deepfake/diffae256",
}


def to_uint8(tensor):
    return (
        tensor[0].permute(1, 2, 0).cpu().numpy() * 255
    ).round().clip(0, 255).astype(np.uint8)


def psnr(a, b):
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--models_path", required=True)
    parser.add_argument("--image_resolution", type=int, default=256)
    parser.add_argument("--val_dir", default="Dataset/celeba_hq/val")
    parser.add_argument("--output_dir", default="my_tests/diffae")
    parser.add_argument("--reference", default="003251", help="Imagem sempre incluída no teste.")
    parser.add_argument("--num_female", type=int, default=3)
    parser.add_argument("--num_male", type=int, default=2)
    parser.add_argument("--attribute", default="Smiling", help="Atributo do CelebA (ex.: Smiling, Young, Male, Eyeglasses).")
    parser.add_argument("--strength", type=float, default=0.3, help="Positivo adiciona o atributo; negativo remove.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.image_resolution not in DIFFAE_DIRS:
        raise ValueError(f"Não há DiffAE para {args.image_resolution}x{args.image_resolution}.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dirs = {name: os.path.join(args.output_dir, name) for name in ["originais", "marcadas", "diffae"]}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # -------------------------
    # Imagens: referência + amostras femininas e masculinas
    # -------------------------
    female = sorted(glob.glob(os.path.join(args.val_dir, "female", "*.jpg")))
    male = sorted(glob.glob(os.path.join(args.val_dir, "male", "*.jpg")))

    reference = [p for p in female + male if os.path.splitext(os.path.basename(p))[0] == args.reference]
    female = [p for p in female if p not in reference]
    male = [p for p in male if p not in reference]

    n_female = args.num_female - sum("female" in p for p in reference)
    n_male = args.num_male - sum("/male/" in p for p in reference)
    paths = reference + random.sample(female, max(n_female, 0)) + random.sample(male, max(n_male, 0))

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

    print(f"Marca: {num_bits} bits | {len(paths)} imagens | DiffAE {args.image_resolution} "
          f"({args.attribute}, strength {args.strength})")

    image_transform = transforms.Compose([
        transforms.Resize((args.image_resolution, args.image_resolution)),
        transforms.ToTensor(),
    ])

    # -------------------------
    # 1. Marcar (salva em PNG, que é a entrada do DiffAE)
    # -------------------------
    names, watermarks = [], {}

    with torch.no_grad():
        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            names.append(name)

            image = image_transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            watermark = torch.randint(0, 2, (1, num_bits), device=device).float()
            watermarks[name] = watermark

            marked, _, _ = watermark_embedding_tensor(image, watermark, embedding_model)

            Image.fromarray(to_uint8(image)).save(os.path.join(dirs["originais"], name + ".png"))
            Image.fromarray(to_uint8(marked)).save(os.path.join(dirs["marcadas"], name + ".png"))

    # -------------------------
    # 2. DiffAE (ambiente conda separado)
    # -------------------------
    print("Executando DiffAE...")

    result = subprocess.run(
        [
            DIFFAE_PYTHON, "diffae_dataset_infer.py",
            "--input_dir", os.path.abspath(dirs["marcadas"]),
            "--output_dir", os.path.abspath(dirs["diffae"]),
            "--attribute", args.attribute,
            "--strength", str(args.strength),
            "--device", "cuda:0",
        ],
        cwd=DIFFAE_DIRS[args.image_resolution],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(result.stderr[-3000:])
        sys.exit("Erro ao executar o DiffAE.")

    # -------------------------
    # 3. Extrair a marca antes e depois do DiffAE
    # -------------------------
    def bit_accuracy(image_array, name):
        tensor = transforms.ToTensor()(Image.fromarray(image_array)).unsqueeze(0).to(device)
        predicted = (extractor_model(tensor) >= 0).float()
        return (predicted == watermarks[name]).float().mean().item()

    rows, tiles = [], []

    with torch.no_grad():
        for name in names:
            original = np.array(Image.open(os.path.join(dirs["originais"], name + ".png")).convert("RGB"))
            marked = np.array(Image.open(os.path.join(dirs["marcadas"], name + ".png")).convert("RGB"))
            attacked = np.array(Image.open(os.path.join(dirs["diffae"], name + ".png")).convert("RGB"))

            if attacked.shape != marked.shape:
                sys.exit(f"DiffAE devolveu {attacked.shape[:2]}, esperado {marked.shape[:2]}.")

            rows.append((
                name,
                bit_accuracy(marked, name),
                bit_accuracy(attacked, name),
                psnr(original, marked),
                psnr(marked, attacked),
            ))
            tiles.append(np.concatenate([original, marked, attacked], axis=1))

    grid_path = os.path.join(args.output_dir, "grid_diffae.png")
    Image.fromarray(np.concatenate(tiles, axis=0)).save(grid_path)

    # -------------------------
    # 4. Resultado
    # -------------------------
    print("\n| Imagem | Bit Accuracy (marcada) | Bit Accuracy (após DiffAE) | PSNR marcada × original | PSNR DiffAE × marcada |")
    print("|---|---|---|---|---|")
    for name, acc_marked, acc_attacked, psnr_marked, psnr_attacked in rows:
        print(f"| {name} | {acc_marked * 100:.2f}% | {acc_attacked * 100:.2f}% | {psnr_marked:.2f} | {psnr_attacked:.2f} |")

    means = np.mean([r[1:] for r in rows], axis=0)
    print(f"| **Média** | {means[0] * 100:.2f}% | {means[1] * 100:.2f}% | {means[2]:.2f} | {means[3]:.2f} |")
    print(f"\nGrid salvo em: {grid_path}")


if __name__ == "__main__":
    main()
