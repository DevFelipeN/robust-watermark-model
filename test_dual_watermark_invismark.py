#!/usr/bin/env python3
"""
test_dual_watermark_invismark.py

Mesmo teste de MARCA DUPLA do test_dual_watermark.py, trocando o FaceSigns
pelo InvisMark (WACV 2025, checkpoint do artigo), sem treinar nada:

    imagem -> marca deste modelo (Y/DWT/DCT, 30 bits)     -> prova a origem
           -> marca InvisMark   (pixels, 100 bits)        -> segunda marca

A deste modelo vem primeiro e a do InvisMark por último, como no teste com o
FaceSigns.

Avalia só a imagem DUPLA (as duas marcas, cnn e depois InvisMark): cada
ataque é aplicado uma vez e as duas marcas são lidas da mesma imagem.

Responde:
    1. a marca deste modelo continua robusta com a do InvisMark por cima?
    2. o InvisMark sobrevive aos ataques comuns e quebra nos deepfakes?
    3. quanto custa em qualidade (PSNR, SSIM, LPIPS: original x dupla)?

Atenção: o InvisMark NÃO foi feito para ser frágil. Em 256x256 ele perde a
marca tanto em deepfakes quanto em JPEG (o resíduo só é ampliado, e vira
baixa frequência, em imagens grandes), então o papel de "detector" aqui é o
que o teste quer verificar, não uma premissa.

Ataques: os 16 do evaluate.py + os 7 deepfakes do evaluate.py, todos nas
--num_images imagens (DiffAE leva ~11 s por imagem com a GPU livre), com
a mesma seed e as mesmas origens do pares_origem.csv do evaluate.py.
As imagens são metade femininas e metade masculinas, como no --limit do
evaluate.py.

Ambiente: invismark (clone do watermark-v8 + bchlib e kornia).

Exemplo:
    python test_dual_watermark_invismark.py --num_images 100
"""

import os
import sys
import math
import random
import shutil
import argparse
import uuid

import lpips
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torchmetrics.functional.image import structural_similarity_index_measure

from attacks import AttackLayer
from deepfake_attack import DeepFakeAttack
from watermark_embedding import WatermarkEmbedding, watermark_embedding_tensor
from watermark_extraction import WatermarkExtraction


INVISMARK_DIR = "/home/cnt-computadores/Área de trabalho/FelipeNascimento/models_watermark_others/InvisMark"
ORIGENS_FACESWAP_DIR = "/home/cnt-computadores/Área de trabalho/FelipeNascimento/models_deepfake/origens_faceswap"

# model.py/configs.py do InvisMark no fim do path: os módulos deste projeto
# (attacks, utils...) continuam tendo prioridade
sys.path.append(INVISMARK_DIR)
sys.path.append(ORIGENS_FACESWAP_DIR)
import configs  # noqa: E402,F401  (necessário para ler a config salva no checkpoint)
import model as invismark_model  # noqa: E402
from pares_origem import gerar_pares, salvar_pares, genero_pelo_caminho  # noqa: E402

IMAGE_SIZE = 256

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
    parser.add_argument("--invismark_ckpt", default=os.path.join(INVISMARK_DIR, "ckpts", "paper.ckpt"))
    parser.add_argument("--celeba_val_dir", default="Dataset/celeba_hq/val")
    parser.add_argument("--output", default="my_tests/marca_dupla_invismark")
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shutil.rmtree(args.output, ignore_errors=True)
    os.makedirs(args.output, exist_ok=True)

    # -------------------------
    # Marca deste modelo
    # -------------------------
    ck = torch.load(args.checkpoint, map_location=device)
    num_bits, grid = ck["num_bits"], ck["watermark_size"]

    embedding = WatermarkEmbedding(grid, IMAGE_SIZE, num_bits=num_bits).to(device).eval()
    embedding.load_state_dict(ck["embedding_model"])
    embedding.alpha = ck["alpha"]

    extractor = WatermarkExtraction(grid, IMAGE_SIZE, use_hh=ck.get("use_hh", False), num_bits=num_bits).to(device).eval()
    extractor.load_state_dict(ck["extraction_model"])

    # -------------------------
    # Marca InvisMark (checkpoint do artigo, 256x256, 100 bits)
    # -------------------------
    state = torch.load(args.invismark_ckpt, map_location="cpu", weights_only=False)
    cfg = state["config"]
    cfg.image_shape = tuple(cfg.image_shape)
    im_encoder = invismark_model.Encoder(cfg).to(device).eval()
    im_decoder = invismark_model.Extractor(cfg).to(device).eval()
    im_encoder.load_state_dict(state["encoder_state_dict"])
    im_decoder.load_state_dict(state["decoder_state_dict"])
    del state

    robust_msg = torch.randint(0, 2, (1, num_bits), device=device).float()

    # O checkpoint do InvisMark só lê mensagens no formato UUID4 (bits 48-51 e
    # 64-65 fixos no treino); UUID gerado da seed para ser reprodutível.
    uid = uuid.UUID(bytes=np.random.RandomState(args.seed).bytes(16), version=4)
    invismark_msg = torch.tensor(
        np.unpackbits(np.frombuffer(uid.bytes, dtype=np.uint8))[:cfg.num_encoded_bits],
        dtype=torch.float32, device=device,
    ).unsqueeze(0)

    def add_cnn(x):
        with torch.no_grad():
            marked, _, _ = watermark_embedding_tensor(x, robust_msg, embedding)
        return quantize(marked)

    def add_invismark(x):
        # igual ao train._encode() do InvisMark: resíduo na resolução do
        # checkpoint, somado na resolução da imagem (aqui as duas são 256)
        x_m11 = x * 2 - 1
        with torch.no_grad():
            x_model = F.interpolate(x_m11, size=cfg.image_shape, mode="bilinear", align_corners=False)
            residual = im_encoder(x_model, invismark_msg) - x_model
            residual = F.interpolate(residual, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return quantize(((x_m11 + residual).clamp(-1, 1) + 1) / 2)

    def read_cnn(x):
        with torch.no_grad():
            return ((extractor(x.clamp(0, 1)) >= 0).float() == robust_msg).float().mean().item()

    def read_invismark(x):
        with torch.no_grad():
            probs = im_decoder(x.clamp(0, 1) * 2 - 1)
        return ((probs > 0.5).float() == invismark_msg).float().mean().item()

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

        marked["dupla"][name] = add_invismark(add_cnn(x))   # cnn primeiro, InvisMark por último

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

    def read_all(key, v, attacked):
        if v in ("cnn", "dupla"):
            results.append((key, v, "cnn", read_cnn(attacked)))
        if v in ("invismark", "dupla"):
            results.append((key, v, "invismark", read_invismark(attacked)))

    for key, attack, params in tqdm(ATTACK_CASES, desc="Ataques"):
        for i, name in enumerate(originals):
            for v in VERSIONS:
                set_seed(args.seed + i)
                with torch.no_grad():
                    attacked, _ = attack_layer(marked[v][name], attack=attack, **params)
                read_all(key, v, attacked.clamp(0, 1))

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
            attacked_zip, _ = deepfake_attack.apply_dataset(
                attack=kind, input_zip=input_zip, output_dir=root, **deepfake_params[kind]
            )
            out_dir = os.path.join(root, kind, "output")   # já descompactado pelo DeepFakeAttack

            for name in names:
                out_path = os.path.join(out_dir, name + ".png")
                if not os.path.exists(out_path):            # rosto não detectado
                    print(f"[AVISO] {kind}/{v}: sem saída para {name}")
                    continue
                attacked = transform(Image.open(out_path).convert("RGB")).unsqueeze(0).to(device)
                read_all(kind, v, attacked)

    # -------------------------
    # Tabelas
    # -------------------------
    df = pd.DataFrame(results, columns=["Attack", "Versao", "Marca", "BitAcc"])
    df.to_csv(os.path.join(args.output, "resultados_por_imagem.csv"), index=False)

    order = [k for k, _, _ in ATTACK_CASES] + DEEPFAKES
    table = df.groupby(["Attack", "Versao", "Marca"])["BitAcc"].mean().unstack(["Marca", "Versao"]) * 100
    table = table.reindex(order)
    table.columns = [f"{m} ({v})" for m, v in table.columns]
    table = table[["cnn (dupla)", "invismark (dupla)"]]
    table.columns = ["CNN (na dupla)", "InvisMark (na dupla)"]
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
    print("CNN deve ficar alta em tudo | um detector ficaria alto nos ataques comuns e ~50% nos deepfakes")
    print("=" * 90)
    print(table.round(2).to_string())

    # -------------------------
    # Exemplo visual (primeira imagem)
    # -------------------------
    name = list(originals)[0]
    x = to_uint8(originals[name]).astype(np.float64)
    tiles = [to_uint8(originals[name])] + [to_uint8(marked[v][name]) for v in VERSIONS]
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
