#!/usr/bin/env python3
"""
test_dual_watermark.py

Protótipo de MARCA DUPLA, sem treinar nada:

    imagem -> marca ROBUSTA (este modelo, Y/DWT/DCT, 30 bits)     -> prova a origem
           -> marca FRÁGIL  (FaceSigns pronto, pixels, 128 bits)   -> prova a integridade

A robusta vem primeiro e a frágil por último (a frágil é a mais sensível; para
a robusta, a frágil é só um ruído pequeno).

Avalia só a imagem DUPLA (as duas marcas, robusta e depois FaceSigns): cada
ataque é aplicado uma vez e as duas marcas são lidas da mesma imagem.

Responde:
    1. a sua marca continua robusta com a do FaceSigns por cima?
    2. a do FaceSigns continua robusta a ataques comuns e frágil a deepfakes
       com a sua por baixo?
    3. quanto custa em qualidade (PSNR, SSIM, LPIPS: original x dupla)?

Ataques: os 16 do evaluate.py + os 7 deepfakes do evaluate.py, todos nas
--num_images imagens (DiffAE leva ~11 s por imagem com a GPU livre), com
a mesma seed e as mesmas origens do pares_origem.csv do evaluate.py.
As imagens são metade femininas e metade masculinas, como no --limit do
evaluate.py. Mesmo teste do test_dual_watermark_invismark.py.

Ambiente: watermark-v8.

Exemplo:
    python test_dual_watermark.py --num_images 100
"""

import os
import sys
import math
import random
import shutil
import argparse

import lpips
import numpy as np
import pandas as pd
import torch

from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torchmetrics.functional.image import structural_similarity_index_measure

from attacks import AttackLayer
from deepfake_attack import DeepFakeAttack
from watermark_embedding import WatermarkEmbedding, watermark_embedding_tensor
from watermark_extraction import WatermarkExtraction


FACESIGNS_DIR = "/home/cnt-computadores/Área de trabalho/FelipeNascimento/models_watermark_others/FaceSigns"
ORIGENS_FACESWAP_DIR = "/home/cnt-computadores/Área de trabalho/FelipeNascimento/models_deepfake/origens_faceswap"

sys.path.append(ORIGENS_FACESWAP_DIR)
from pares_origem import gerar_pares, salvar_pares, genero_pelo_caminho  # noqa: E402

IMAGE_SIZE = 256        # o FaceSigns só aceita 256x256

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
    ("dropout_forte50",    "dropout",     {"p": 0.5}),
    ("resize",             "resize",      {"scale": 0.5}),
    ("resize_forte",       "resize",      {"scale": 0.3}),
]

DEEPFAKES = ["diffae", "blendface", "simswap", "inswapper", "stargan", "hisd", "liveportrait"]

VERSIONS = ["dupla"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_images(root):
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)


def limit_by_gender(paths, limit):
    """N imagens, metade de cada gênero (igual ao --limit do evaluate.py)."""
    female = [p for p in paths if os.path.basename(os.path.dirname(p)) == "female"]
    male = [p for p in paths if os.path.basename(os.path.dirname(p)) == "male"]
    n_male = limit // 2
    return sorted(female[:limit - n_male] + male[:n_male])


def to_uint8(x_01):
    return (x_01[0].permute(1, 2, 0).detach().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)


def quantize(x_01):
    """Simula salvar em PNG de 8 bits."""
    return torch.round(x_01.clamp(0, 1) * 255) / 255


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="output/checkpoints/watermark_Y_DCT_DWT_07102026_10:52:58_final.pth")
    parser.add_argument("--celeba_val_dir", default="Dataset/celeba_hq/val")
    parser.add_argument("--output", default="my_tests/marca_dupla_facesigns")
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shutil.rmtree(args.output, ignore_errors=True)
    os.makedirs(args.output, exist_ok=True)

    # -------------------------
    # Marca robusta (este modelo)
    # -------------------------
    ck = torch.load(args.checkpoint, map_location=device)
    num_bits, grid = ck["num_bits"], ck["watermark_size"]

    embedding = WatermarkEmbedding(grid, IMAGE_SIZE, num_bits=num_bits).to(device).eval()
    embedding.load_state_dict(ck["embedding_model"])
    embedding.alpha = ck["alpha"]

    extractor = WatermarkExtraction(grid, IMAGE_SIZE, use_hh=ck.get("use_hh", False), num_bits=num_bits).to(device).eval()
    extractor.load_state_dict(ck["extraction_model"])

    # -------------------------
    # Marca frágil (FaceSigns, TorchScript)
    # -------------------------
    fs_encoder = torch.jit.load(os.path.join(FACESIGNS_DIR, "encoder_model.pth"), map_location=device).eval()
    fs_decoder = torch.jit.load(os.path.join(FACESIGNS_DIR, "decoder_model.pth"), map_location=device).eval()

    robust_msg = torch.randint(0, 2, (1, num_bits), device=device).float()
    fragile_msg = torch.randint(0, 2, (1, 128), device=device)

    def add_robust(x):
        with torch.no_grad():
            marked, _, _ = watermark_embedding_tensor(x, robust_msg, embedding)
        return quantize(marked)

    def add_fragile(x):
        with torch.no_grad():
            marked, _ = fs_encoder(x, fragile_msg)
        return quantize(marked)

    def read_robust(x):
        with torch.no_grad():
            return ((extractor(x.clamp(0, 1)) >= 0).float() == robust_msg).float().mean().item()

    def read_fragile(x):
        with torch.no_grad():
            logits, _ = fs_decoder(x.clamp(0, 1))
        return ((torch.sigmoid(logits) > 0.5).long() == fragile_msg).float().mean().item()

    lpips_metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    attack_layer = AttackLayer().to(device).eval()
    deepfake_attack = DeepFakeAttack()

    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    images = limit_by_gender(find_images(args.celeba_val_dir), args.num_images)

    # mesmas origens dos face swaps / motoras do LivePortrait do evaluate.py
    pairs_csv = os.path.join(args.output, "pares_origem.csv")
    salvar_pares(
        gerar_pares([(os.path.splitext(os.path.basename(p))[0], genero_pelo_caminho(p)) for p in images]),
        pairs_csv
    )

    deepfake_params = {
        "diffae": {"attribute": "Young", "strength": 0.3, "image_resolution": IMAGE_SIZE},
        "blendface": {"pairs_csv": pairs_csv},
        "simswap": {"pairs_csv": pairs_csv},
        "inswapper": {"pairs_csv": pairs_csv},
        "stargan": {"attribute": "Male"},
        "hisd": {"tag": "Eyeglasses", "attribute": "with", "seed": args.seed},
        "liveportrait": {"pairs_csv": pairs_csv},
    }

    # -------------------------
    # Marcar as três versões
    # -------------------------
    marked = {v: {} for v in VERSIONS}
    originals = {}
    quality = {v: [] for v in VERSIONS}

    for path in tqdm(images, desc="Marcando"):
        name = os.path.splitext(os.path.basename(path))[0]
        x = quantize(transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device))
        originals[name] = x

        marked["dupla"][name] = add_fragile(add_robust(x))   # robusta primeiro, frágil por último

        for v in VERSIONS:
            m = marked[v][name]
            o_np, m_np = to_uint8(x).astype(np.float64), to_uint8(m).astype(np.float64)
            psnr = 10 * math.log10(255 ** 2 / max(np.mean((o_np - m_np) ** 2), 1e-10))
            with torch.no_grad():
                ssim = structural_similarity_index_measure(x, m).item()
                lp = lpips_metric(x * 2 - 1, m * 2 - 1).mean().item()
            quality[v].append((psnr, ssim, lp))

    # -------------------------
    # Ataques comuns (mesma aleatoriedade em todas as versões)
    # -------------------------
    results = []   # (ataque, versão, marca, bit acc)

    for key, attack, params in tqdm(ATTACK_CASES, desc="Ataques"):
        for i, name in enumerate(originals):
            for v in VERSIONS:
                set_seed(args.seed + i)
                with torch.no_grad():
                    attacked, _ = attack_layer(marked[v][name], attack=attack, **params)
                attacked = attacked.clamp(0, 1)

                if v in ("robusta", "dupla"):
                    results.append((key, v, "robusta", read_robust(attacked)))
                if v in ("fragil", "dupla"):
                    results.append((key, v, "fragil", read_fragile(attacked)))

    # -------------------------
    # Deepfakes (mesmo fluxo do evaluate.py: ZIP -> modelo externo -> ZIP)
    # -------------------------
    for kind in DEEPFAKES:
        names = list(originals)
        for v in VERSIONS:
            root = os.path.join(args.output, "deepfakes", v)
            in_dir = os.path.join(root, f"{kind}_entrada")
            shutil.rmtree(in_dir, ignore_errors=True)
            os.makedirs(in_dir)
            for name in names:
                Image.fromarray(to_uint8(marked[v][name])).save(os.path.join(in_dir, name + ".png"))
            input_zip = shutil.make_archive(in_dir, "zip", in_dir)

            print(f"{kind}: versão {v} ({len(names)} imagens)...")
            deepfake_attack.apply_dataset(
                attack=kind, input_zip=input_zip, output_dir=root, **deepfake_params[kind]
            )
            out_dir = os.path.join(root, kind, "output")   # já descompactado pelo DeepFakeAttack

            for name in names:
                out_path = os.path.join(out_dir, name + ".png")
                if not os.path.exists(out_path):            # rosto não detectado
                    print(f"[AVISO] {kind}/{v}: sem saída para {name}")
                    continue
                attacked = transform(Image.open(out_path).convert("RGB")).unsqueeze(0).to(device)
                if v in ("robusta", "dupla"):
                    results.append((kind, v, "robusta", read_robust(attacked)))
                if v in ("fragil", "dupla"):
                    results.append((kind, v, "fragil", read_fragile(attacked)))

    # -------------------------
    # Tabelas
    # -------------------------
    df = pd.DataFrame(results, columns=["Attack", "Versao", "Marca", "BitAcc"])
    df.to_csv(os.path.join(args.output, "resultados_por_imagem.csv"), index=False)

    order = [k for k, _, _ in ATTACK_CASES] + DEEPFAKES
    table = df.groupby(["Attack", "Versao", "Marca"])["BitAcc"].mean().unstack(["Marca", "Versao"]) * 100
    table = table.reindex(order)
    table.columns = [f"{m} ({v})" for m, v in table.columns]
    table = table[["robusta (dupla)", "fragil (dupla)"]]
    table.columns = ["Robusta (na dupla)", "Frágil (na dupla)"]
    table.to_csv(os.path.join(args.output, "bit_accuracy.csv"))

    q = pd.DataFrame(
        {v: np.mean(quality[v], axis=0) for v in VERSIONS}, index=["PSNR", "SSIM", "LPIPS"]
    ).T
    q.to_csv(os.path.join(args.output, "qualidade.csv"))

    print("\n" + "=" * 90)
    print(f"QUALIDADE (original x marcada, {len(originals)} imagens)")
    print("=" * 90)
    print(q.round(4).to_string())

    print("\n" + "=" * 90)
    print(f"BIT ACCURACY (%) — imagem dupla, {len(originals)} imagens em todos os ataques")
    print("Robusta deve ficar alta em tudo | Frágil deve ficar alta nos ataques comuns e ~50% nos deepfakes")
    print("=" * 90)
    print(table.round(2).to_string())

    # -------------------------
    # Exemplo visual (primeira imagem)
    # -------------------------
    name = list(originals)[0]
    x = to_uint8(originals[name]).astype(np.float64)
    tiles = [to_uint8(originals[name])]
    for v in VERSIONS:
        m = to_uint8(marked[v][name])
        tiles.append(m)
    diffs = [np.zeros_like(tiles[0])] + [
        np.clip(128 + (to_uint8(marked[v][name]).astype(np.float64) - x) * 10, 0, 255).astype(np.uint8) for v in VERSIONS
    ]
    grid_img = np.concatenate([np.concatenate(tiles, 1), np.concatenate(diffs, 1)], 0)
    Image.fromarray(grid_img).save(os.path.join(args.output, "exemplo.png"))

    print(f"\nExemplo visual (original | dupla; abaixo, diferença x10): "
          f"{os.path.join(args.output, 'exemplo.png')}")
    print(f"CSVs em: {args.output}")


if __name__ == "__main__":
    main()
