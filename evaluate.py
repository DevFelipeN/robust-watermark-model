import os
import sys
import math
import random
import shutil
import argparse

import cv2
import numpy as np
import pandas as pd
import torch

from PIL import Image
from torchvision import transforms

from utils import (block_idct2d, inverse_arnold_transform)

from watermark_embedding import (WatermarkEmbedding, watermark_embedding_tensor)
from watermark_extraction import WatermarkExtraction

from watermark_extraction import WatermarkExtraction
from attacks import AttackLayer
from deepfake_attack import DeepFakeAttack, DEEPFAKE_ROOT

# 20 identidades de origem usadas pelos deepfakes de troca de rosto
ORIGENS_FACESWAP_DIR = os.path.join(DEEPFAKE_ROOT, "origens_faceswap")

from torchmetrics.functional.image import structural_similarity_index_measure
from tabulate import tabulate
from tqdm import tqdm
import lpips
from DISTS_pytorch import DISTS


# ==========================================
# FIND IMAGES
# ==========================================
def find_images(root):

    paths = []

    for dirpath, _, filenames in os.walk(root):

        for f in filenames:

            if f.lower().endswith((".jpg", ".jpeg", ".png")):

                paths.append(
                    os.path.join(dirpath, f)
                )

    return sorted(paths)


# ==========================================
# MAIN
# ==========================================
def main():

    parser = argparse.ArgumentParser()

    # -------------------------
    # Arguments
    # -------------------------
    parser.add_argument("--models_path", required=True)
    parser.add_argument("--celeba_val_dir", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--image_resolution", type=int, default=128)
    parser.add_argument("--watermark_size", type=int, default=None, help="Opcional: o tamanho da marca é lido do checkpoint. Se informado, só é conferido.")
    parser.add_argument("--deepfake", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--arnold_iterations", type=int, default=0)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help="Seed da marca e dos ataques aleatórios (ruído, dropout, S&P).")
    parser.add_argument("--limit", type=int, default=None, help="Usar só N imagens, metade de cada gênero (inclusive nos deepfakes).")
    parser.add_argument("--alpha", type=float, default=None, help="Opcional: força da marca (padrão: o alpha salvo no checkpoint). Na opção 3, 5.365 dá o mesmo PSNR (47,2 dB) dos outros modelos.")

    args = parser.parse_args()

    # Sem seed, cada avaliação usa outra marca e outros ruídos,
    # e duas execuções do mesmo modelo dão números diferentes
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # -------------------------
    # CUDA
    # -------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # -------------------------
    # Métricas perceptuais
    # -------------------------
    lpips_metric = lpips.LPIPS(net="alex").to(device)
    lpips_metric.eval()

    dists_metric = DISTS().to(device)
    dists_metric.eval()

    # -------------------------
    # Output
    # -------------------------
    os.makedirs(args.output, exist_ok=True)

    watermarked_dir = os.path.join(args.output, "watermarked_images")

    # limpa imagens de execuções anteriores (senão vão todas para os deepfakes)
    shutil.rmtree(watermarked_dir, ignore_errors=True)
    os.makedirs(watermarked_dir, exist_ok=True)

    # -------------------------
    # Checkpoint (lido antes da marca: define se ela é quadrada ou vetor)
    # -------------------------
    checkpoint = torch.load(args.models_path, map_location=device)

    # Tamanho da marca vem do checkpoint:
    # - novos: num_bits bits, mapa interno grid_size x grid_size
    # - antigos (sem num_bits): marca quadrada grid_size x grid_size,
    #   com grid_size deduzido do fator de ampliação da primeira camada
    num_bits = checkpoint.get("num_bits", 0)

    if num_bits:
        grid_size = checkpoint["watermark_size"]
        marca_bits = num_bits
        print(f"Marca: {num_bits} bits (mapa interno {grid_size}x{grid_size})")
    else:
        upsample = checkpoint["embedding_model"]["watermark_encoder.0.weight"].shape[-1]
        grid_size = (args.image_resolution // 2) // upsample
        marca_bits = grid_size ** 2
        print(f"Marca (modelo antigo, quadrada): {grid_size}x{grid_size} = {marca_bits} bits")

    if args.watermark_size is not None and args.watermark_size not in (marca_bits, grid_size):
        raise ValueError(
            f"--watermark_size {args.watermark_size} não corresponde ao modelo "
            f"({marca_bits} bits). O argumento pode ser omitido."
        )

    # -------------------------
    # Watermark (fixa para todo o dataset)
    # -------------------------
    watermark_shape = (
        (1, num_bits) if num_bits
        else (1, 1, grid_size, grid_size)
    )

    watermark_tensor = torch.randint(
        0,
        2,
        watermark_shape,
        device=device,
        dtype=torch.float32,
    )

    watermark_original_binary = (
        (watermark_tensor[0] if num_bits else watermark_tensor[0, 0])
        .detach()
        .cpu()
        .numpy()
        > 0.5
    ).astype(np.uint8)

    selected_wm = (
        (watermark_tensor[0].reshape(1, -1) if num_bits else watermark_tensor[0, 0]).cpu().numpy() * 255
    ).round().astype(np.uint8)

    watermark_path = os.path.join(args.output, "watermark_dataset.png")

    cv2.imwrite(watermark_path, selected_wm)

    # -------------------------
    # Transforms
    # -------------------------
    image_transform = transforms.Compose([
        transforms.Resize(
            (args.image_resolution, args.image_resolution)
        ),
        transforms.ToTensor(),
    ])

    # -------------------------
    # Load models
    # -------------------------
    embedding_model = WatermarkEmbedding(
        watermark_size=grid_size,
        image_resolution=args.image_resolution,
        num_bits=num_bits or None,
    ).to(device)

    extractor_model = WatermarkExtraction(
        watermark_size=grid_size,
        image_resolution=args.image_resolution,
        use_hh=checkpoint.get("use_hh", False),
        num_bits=num_bits or None,
    ).to(device)

    embedding_model.load_state_dict(checkpoint["embedding_model"])
    extractor_model.load_state_dict(checkpoint["extraction_model"])
    embedding_model.alpha = args.alpha if args.alpha is not None else checkpoint["alpha"]
    print(f"Alpha: {embedding_model.alpha}")

    attack_layer = AttackLayer().to(device)
    deepfake_attack = DeepFakeAttack()

    embedding_model.eval()
    extractor_model.eval()
    attack_layer.eval()

    attack_layer = AttackLayer().to(device)
    deepfake_attack = DeepFakeAttack()

    for p in embedding_model.parameters():
        print(p.mean(), p.std())
        break

    print("Modelos carregados.")

    # -------------------------
    # CelebA-HQ images
    # -------------------------
    celeba_images = find_images(args.celeba_val_dir)

    if len(celeba_images) == 0:
        raise ValueError(
            "Nenhuma imagem encontrada na pasta de validação."
        )
    else:
        print(f"Total de imagens: {len(celeba_images)}")

    if args.limit is not None:
        # metade de cada gênero (pastas female/ e male/); se o limite for
        # ímpar, a imagem extra é feminina
        female = [p for p in celeba_images if os.path.basename(os.path.dirname(p)) == "female"]
        male = [p for p in celeba_images if os.path.basename(os.path.dirname(p)) == "male"]

        if not female or not male:
            raise ValueError("--limit por gênero precisa das subpastas female/ e male/ na validação.")

        n_male = args.limit // 2
        n_female = args.limit - n_male
        celeba_images = sorted(female[:n_female] + male[:n_male])

        print(f"Limite: {n_female} femininas + {n_male} masculinas")

    print(f"Total de imagens utilizadas no teste: {len(celeba_images)}")

    # -------------------------
    # Origens da troca de rosto (uma por imagem, mesmo gênero, sorteio fixo)
    # -------------------------
    # só com deepfakes: o sorteio precisa do gênero (pastas female/ e male/),
    # que outros conjuntos (ex.: LFW) não têm
    pairs_csv = os.path.join(args.output, "pares_origem.csv")
    if args.deepfake:
        sys.path.insert(0, ORIGENS_FACESWAP_DIR)
        from pares_origem import gerar_pares, salvar_pares, genero_pelo_caminho
        salvar_pares(
            gerar_pares([(os.path.splitext(os.path.basename(p))[0], genero_pelo_caminho(p)) for p in celeba_images]),
            pairs_csv
        )
        print(f"Pares alvo -> origem (troca de rosto): {pairs_csv}")

    # -------------------------
    # Ataques avaliados
    # -------------------------
    # (nome no relatório, ataque do AttackLayer, parâmetros)
    # Agrupados por tipo, do mais fraco ao mais forte. Os sem "_forte"
    # mantêm os nomes e parâmetros antigos (comparáveis com resultados
    # anteriores); os "_forte" são versões mais agressivas do mesmo ataque.
    attack_cases = [
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

    attacks = [key for key, _, _ in attack_cases]
    attack_names = {key: name for key, name, _ in attack_cases}
    attack_params = {key: params for key, _, params in attack_cases}

    attacks_deepfake = [
        "diffae",
        "blendface",
        "simswap",
        "inswapper",
        "stargan",
        "hisd",
        "liveportrait"
    ]

    attack_params_deepfake = {
        "diffae": {
            "attribute": "Young",
            "strength": 0.3,
            "image_resolution": args.image_resolution   # models_deepfake/diffae128 ou diffae256
        },

        # Troca de rosto (BlendFace; SimSwap: recorte 224, máscara por
        # segmentação; InSwapper/Roop: recorte 128, máscara retangular).
        # Cada imagem recebe uma das 20 identidades de origem
        # (models_deepfake/origens_faceswap), sorteada de forma fixa entre as
        # 10 do mesmo gênero; os pares ficam em <output>/pares_origem.csv.
        "blendface": {"pairs_csv": pairs_csv},
        "simswap": {"pairs_csv": pairs_csv},
        "inswapper": {"pairs_csv": pairs_csv},

        # Edição de atributos por GAN (StarGAN v1, CelebA 256): mesmo protocolo
        # do SepMark, atributos reais da imagem com Male invertido.
        "stargan": {"attribute": "Male"},

        # Edição de atributos por GAN de alta qualidade (HiSD, CVPR 2021,
        # CelebA-HQ 256): adiciona óculos, mantendo o resto da imagem.
        "hisd": {"tag": "Eyeglasses", "attribute": "with", "seed": args.seed},

        # Reencenação (LivePortrait, guiado por imagem): o alvo recebe a pose
        # e a expressão da mesma origem dos face swaps (movimento relativo) e o
        # rosto é colado de volta na foto original.
        "liveportrait": {"pairs_csv": pairs_csv},
    }

    # Acumuladores de métricas por ataque
    all_attacks = attacks + attacks_deepfake

    metrics = {
        attack: {
            "ber": [],
            "bitacc": [],
            "accuracy": [],
            "exact": [],
            "psnr": [],
            "ssim": [],
            "lpips": [],
            "dists": [],
        }
        for attack in all_attacks
    }

    image_results = []  

    # ==========================================
    # LOOP DAS IMAGENS
    # ==========================================
    for image_path in tqdm(celeba_images):

        image = Image.open(image_path).convert("RGB")

        # Imagem ORIGINAL (sem marca) — referência para PSNR/SSIM
        image_tensor = (
            image_transform(image)
            .unsqueeze(0)
            .to(device)
        )

        # -------------------------
        # Embedding
        # -------------------------
        watermarked, target, modification = watermark_embedding_tensor(
            image_tensor=image_tensor,
            watermark_image=watermark_tensor,
            embedding_model=embedding_model,
            arnold_iterations=args.arnold_iterations,
        )

        # ==========================================
        # Salvar imagem watermarkada (dataset)
        # ==========================================
        watermarked_np = (
            watermarked[0]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )

        watermarked_np = np.clip(
            np.round(watermarked_np * 255), 0, 255
        ).astype(np.uint8)

        #output_name = os.path.basename(image_path)
        output_name = (
            os.path.splitext(
                os.path.basename(image_path)
            )[0]
            + ".png"
        )

        cv2.imwrite(
            os.path.join(watermarked_dir, output_name),
            cv2.cvtColor(watermarked_np, cv2.COLOR_RGB2BGR),
        )

        # ==========================================
        # Avaliação: ataques sobre a imagem watermarkada
        # ==========================================
        original_np = (
            image_tensor[0]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
            * 255
        ).round().astype(np.uint8)

        for attack in attacks:

            # ----------------------------------------------
            # Aplica ataque sobre a imagem MARCADA
            # (attack="none" -> apenas a própria imagem marcada)
            # ----------------------------------------------
            params = attack_params.get(attack, {})

            attacked, attack_info = attack_layer(
                watermarked,
                attack=attack_names[attack],
                **params
            )

            # ----------------------------------------------
            # Extrai marca da imagem atacada
            # ----------------------------------------------
            with torch.no_grad():

                pred = extractor_model(attacked)
                pred = (torch.sigmoid(pred) > 0.5).float()

            watermark_extracted = (
                (pred[0] if num_bits else pred[0, 0])
                .detach()
                .cpu()
                .numpy()
            )

            if args.arnold_iterations > 0 and not num_bits:
                watermark_extracted = inverse_arnold_transform(
                    watermark_extracted,
                    args.arnold_iterations,
                )

            watermark_extracted_binary = (
                watermark_extracted > 0.5
            ).astype(np.uint8)

            # ----------------------------------------------
            # BER / Bit Accuracy
            # ----------------------------------------------
            bit_errors = np.sum(
                watermark_original_binary != watermark_extracted_binary
            )

            total_bits = watermark_original_binary.size

            ber = bit_errors / total_bits
            bit_acc = 1.0 - ber
            accuracy = bit_acc * 100.0

            # ----------------------------------------------
            # PSNR: imagem ORIGINAL vs. marcada (ou marcada+atacada)
            # ----------------------------------------------
            attacked_np = (
                attacked[0]
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
                * 255
            ).round().astype(np.uint8)

            mse = np.mean(
                (
                    original_np.astype(np.float32)
                    - attacked_np.astype(np.float32)
                ) ** 2
            )

            if mse == 0:
                psnr = float("inf")
            else:
                psnr = 10 * math.log10((255.0 ** 2) / mse)

            # ----------------------------------------------
            # SSIM: imagem ORIGINAL vs. marcada (ou marcada+atacada)
            # ----------------------------------------------
            ssim_value = structural_similarity_index_measure(
                image_tensor, attacked
            ).item()

            # ----------------------------------------------
            # LPIPS e DISTS
            # ----------------------------------------------
            with torch.no_grad():
                # LPIPS espera imagens no intervalo [-1, 1]
                original_lpips = image_tensor * 2.0 - 1.0
                attacked_lpips = attacked * 2.0 - 1.0

                lpips_value = lpips_metric(
                    original_lpips, attacked_lpips
                ).mean().item()

                # DISTS espera imagens no intervalo [0, 1]
                dists_value = dists_metric(
                    image_tensor, attacked
                ).mean().item()

            # ----------------------------------------------
            # Armazena métricas
            # ----------------------------------------------
            metrics[attack]["ber"].append(ber)
            metrics[attack]["bitacc"].append(bit_acc)
            metrics[attack]["accuracy"].append(accuracy)
            # marca recuperada por inteiro (nenhum bit errado)
            metrics[attack]["exact"].append(float(bit_errors == 0))
            metrics[attack]["psnr"].append(psnr)
            metrics[attack]["ssim"].append(ssim_value)
            metrics[attack]["lpips"].append(lpips_value)
            metrics[attack]["dists"].append(dists_value)

            image_results.append({

                "Image": os.path.basename(image_path),

                "Attack": attack,

                "BER": ber,

                "Bit Accuracy": bit_acc,

                "Accuracy": accuracy,


                "Exact Match": int(bit_errors == 0),

                "PSNR": psnr,

                "SSIM": ssim_value,

                "LPIPS": lpips_value,

                "DISTS": dists_value

            })

    # ==========================================
    # Compactar dataset watermarkado
    # ==========================================
    print("Compactando dataset watermarcado...")

    zip_path = os.path.join(args.output, "watermarked_dataset")

    shutil.make_archive(zip_path, "zip", watermarked_dir)

    watermarked_zip = zip_path + ".zip"

    print(f"Dataset watermarkado: {watermarked_zip}")

    if args.deepfake:

        #==============================================
        # Iniciar ataques maliguinos (Deepfakes)
        #===============================================

        for attack in attacks_deepfake:
            parameters = attack_params_deepfake.get(attack, {})
            attacked_zip, attack_info = deepfake_attack.apply_dataset(
                attack=attack,
                input_zip=watermarked_zip,
                output_dir=args.output,
                **parameters
            )

            attacked_dir = os.path.join(
                args.output,
                f"{attack}_images"
            )

            shutil.rmtree(attacked_dir, ignore_errors=True)

            os.makedirs(
                attacked_dir,
                exist_ok=True
            )

            shutil.unpack_archive(
                attacked_zip,
                attacked_dir
            )

                    # =====================================================
            # Avaliar imagens produzidas pelo DeepFake
            # =====================================================

            for attacked_path in find_images(attacked_dir):

                filename = os.path.basename(attacked_path)

                # -------------------------------------------------
                # Localizar imagem ORIGINAL correspondente
                # -------------------------------------------------

                original_path = None

                for path in celeba_images:
                    if os.path.splitext(os.path.basename(path))[0] == \
                    os.path.splitext(filename)[0]:
                        original_path = path
                        break

                if original_path is None:
                    print(
                        f"[AVISO] Original não encontrada para: {filename}"
                    )
                    continue

                # -------------------------------------------------
                # Carregar original
                # -------------------------------------------------

                original_image = (
                    Image.open(original_path)
                    .convert("RGB")
                )

                original_tensor = (
                    image_transform(original_image)
                    .unsqueeze(0)
                    .to(device)
                )

                # -------------------------------------------------
                # Carregar imagem atacada
                # -------------------------------------------------

                attacked_image = (
                    Image.open(attacked_path)
                    .convert("RGB")
                )

                attacked = (
                    image_transform(attacked_image)
                    .unsqueeze(0)
                    .to(device)
                )

                # -------------------------------------------------
                # Extrair watermark
                # -------------------------------------------------

                with torch.no_grad():

                    pred = extractor_model(attacked)

                    pred = (
                        torch.sigmoid(pred) > 0.5
                    ).float()

                watermark_extracted = (
                    (pred[0] if num_bits else pred[0, 0])
                    .detach()
                    .cpu()
                    .numpy()
                )

                # -------------------------------------------------
                # Inverse Arnold
                # -------------------------------------------------

                if args.arnold_iterations > 0 and not num_bits:

                    watermark_extracted = (
                        inverse_arnold_transform(
                            watermark_extracted,
                            args.arnold_iterations
                        )
                    )

                watermark_extracted_binary = (
                    watermark_extracted > 0.5
                ).astype(np.uint8)

                # -------------------------------------------------
                # BER / Bit Accuracy
                # -------------------------------------------------

                bit_errors = np.sum(
                    watermark_original_binary
                    != watermark_extracted_binary
                )

                total_bits = watermark_original_binary.size

                ber = bit_errors / total_bits

                bit_acc = 1.0 - ber

                accuracy = bit_acc * 100.0

                # -------------------------------------------------
                # Converter para numpy
                # -------------------------------------------------

                original_np = (
                    original_tensor[0]
                    .permute(1, 2, 0)
                    .detach()
                    .cpu()
                    .numpy()
                    * 255
                ).round().astype(np.uint8)

                attacked_np = (
                    attacked[0]
                    .permute(1, 2, 0)
                    .detach()
                    .cpu()
                    .numpy()
                    * 255
                ).round().astype(np.uint8)

                # -------------------------------------------------
                # PSNR
                # -------------------------------------------------

                mse = np.mean(
                    (
                        original_np.astype(np.float32)
                        - attacked_np.astype(np.float32)
                    ) ** 2
                )

                if mse == 0:
                    psnr = float("inf")
                else:
                    psnr = 10 * math.log10(
                        (255.0 ** 2) / mse
                    )

                # -------------------------------------------------
                # SSIM
                # -------------------------------------------------

                ssim_value = (
                    structural_similarity_index_measure(
                        original_tensor,
                        attacked
                    ).item()
                )

                # -------------------------------------------------
                # LPIPS / DISTS
                # -------------------------------------------------

                with torch.no_grad():

                    original_lpips = (
                        original_tensor * 2.0 - 1.0
                    )

                    attacked_lpips = (
                        attacked * 2.0 - 1.0
                    )

                    lpips_value = (
                        lpips_metric(
                            original_lpips,
                            attacked_lpips
                        )
                        .mean()
                        .item()
                    )

                    dists_value = (
                        dists_metric(
                            original_tensor,
                            attacked
                        )
                        .mean()
                        .item()
                    )

                # -------------------------------------------------
                # Armazenar métricas
                # -------------------------------------------------

                metrics[attack]["ber"].append(ber)

                metrics[attack]["bitacc"].append(
                    bit_acc
                )

                metrics[attack]["accuracy"].append(
                    accuracy
                )

                metrics[attack]["exact"].append(
                    float(bit_errors == 0)
                )

                metrics[attack]["psnr"].append(
                    psnr
                )

                metrics[attack]["ssim"].append(
                    ssim_value
                )

                metrics[attack]["lpips"].append(
                    lpips_value
                )

                metrics[attack]["dists"].append(
                    dists_value
                )

                # -------------------------------------------------
                # Resultado por imagem
                # -------------------------------------------------

                image_results.append({

                    "Image": filename,

                    "Attack": attack,

                    "BER": ber,

                    "Bit Accuracy": bit_acc,

                    "Accuracy": accuracy,


                    "Exact Match": int(bit_errors == 0),

                    "PSNR": psnr,

                    "SSIM": ssim_value,

                    "LPIPS": lpips_value,

                    "DISTS": dists_value
                })
        


    # ==========================================
    # Tabela final de resultados
    # ==========================================
    summary = []

    for attack in all_attacks:

        m = metrics[attack]

        # ataque que não rodou (ex.: diffae com --deepfake false)
        if len(m["bitacc"]) == 0:
            continue

        if attack in attacks:
            params = attack_params.get(attack, {})
        else:
            params = attack_params_deepfake.get(attack, {})

        # tipo do ataque (ex.: "jpeg_test_forte30" -> "jpeg_test")
        kind = attack_names.get(attack, attack)

        if kind == "none":
            parameter = "-"

        elif kind == "jpeg":
            parameter = f"Q={params.get('quality')}"
        
        elif kind == "jpeg_test":
            parameter = f"Q={params.get('quality')}"

        elif kind == "blur":
            parameter = f"sigma={params.get('sigma')}"

        elif kind == "median_blur":
            parameter = f"kernel={params.get('kernel')}"

        elif kind == "noise":
            parameter = f"std={params.get('std')}"

        elif kind == "salt_pepper":
            parameter = f"amount={params.get('amount')}"

        elif kind == "dropout":
            parameter = f"p={params.get('p')}"

        elif kind == "resize":
            parameter = f"scale={params.get('scale')}"
        
        elif kind == "diffae":
            parameter = f"attribute={params.get('attribute')}"

        elif kind == "stargan":
            parameter = f"attribute={params.get('attribute')}"

        elif kind == "liveportrait":
            parameter = "20 motoras (mesmo gênero)"

        elif kind == "hisd":
            parameter = f"{params.get('tag')}={params.get('attribute')}"

        elif kind in ("blendface", "simswap", "inswapper"):
            parameter = "20 origens (mesmo gênero)"

        else:
            parameter = "-"


        summary.append({
            "Attack": attack,
            "Parameter": parameter,

            "BER": np.mean(m["ber"]),
            "BER Std": np.std(m["ber"]),

            "Bit Accuracy": np.mean(m["bitacc"]),

            "Accuracy (%)": np.mean(m["accuracy"]),

            "PSNR": np.mean(m["psnr"]),

            "SSIM": np.mean(m["ssim"]),

            "LPIPS": np.mean(m["lpips"]),

            "DISTS": np.mean(m["dists"]),
        })

        print(
            f"{attack:15s}"
            f" BER={np.mean(m['ber']):.4f}"
            f" BitAcc={np.mean(m['bitacc']):.4f}"
            f" Acc={np.mean(m['accuracy']):.2f}%"
            f" Exact={np.mean(m['exact']) * 100:.2f}%"
            f" PSNR={np.mean(m['psnr']):.2f}"
            f" SSIM={np.mean(m['ssim']):.4f}"
            f" LPIPS={np.mean(m['lpips']):.4f}"
            f" DISTS={np.mean(m['dists']):.4f}"
        )

    df = pd.DataFrame(summary)

    df_display = df.copy()
    df_display["BER"] = df_display["BER"].map(lambda x: f"{x:.2f}")
    df_display["BER Std"] = df_display["BER Std"].map(lambda x: f"{x:.2f}")
    df_display["Bit Accuracy"] = df_display["Bit Accuracy"].map(lambda x: f"{x:.4f}")
    df_display["Accuracy (%)"] = df_display["Accuracy (%)"].map(lambda x: f"{x:.2f}%")
    df_display["PSNR"] = df_display["PSNR"].map(lambda x: f"{x:.2f}")
    df_display["SSIM"] = df_display["SSIM"].map(lambda x: f"{x:.4f}")
    df_display["LPIPS"] = df_display["LPIPS"].map(lambda x: f"{x:.4f}")
    df_display["DISTS"] = df_display["DISTS"].map(lambda x: f"{x:.4f}")

    print("\n" + "=" * 100)
    print("RESULTADOS FINAIS")
    print("=" * 100)

    print(tabulate(
        df_display,
        headers="keys",
        tablefmt="grid",
        showindex=False,
    ))

    csv_path = os.path.join(args.output, "media_per_attack.csv")
    df.to_csv(csv_path, index=False)

    df_images = pd.DataFrame(image_results)

    csv_images = os.path.join(
        args.output,
        "results_per_image.csv"
    )

    df_images.to_csv(csv_images, index=False)

    print(f"Resultados por imagem salvos em:\n{csv_images}")

    print("\n===================================")
    print("Avaliação finalizada.")
    print(f"Resultados salvos em:\n{csv_path}")


if __name__ == "__main__":
    main()