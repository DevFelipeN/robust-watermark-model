import os
import math
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from watermark_embedding import (
    WatermarkEmbedding,
    watermark_embedding_tensor,
)

from torchmetrics.functional.image import structural_similarity_index_measure
from tabulate import tabulate
from tqdm import tqdm
import lpips
from DISTS_pytorch import DISTS


# ============================================================
# FIND IMAGES
# ============================================================
def find_images(root):
    paths = []

    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                paths.append(os.path.join(dirpath, f))

    return sorted(paths)


# ============================================================
# MAIN
# ============================================================
def main():

    parser = argparse.ArgumentParser(
        description="Avalia a qualidade visual do watermark no LFW."
    )

    parser.add_argument("--models_path", required=True)
    parser.add_argument("--lfw_dir", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--image_resolution", type=int, default=128)
    parser.add_argument("--watermark_size", type=int, default=16)
    parser.add_argument("--arnold_iterations", type=int, default=0)
    parser.add_argument("--cuda", type=int, default=0)

    # Opcional: testar somente parte do LFW
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Quantidade máxima de imagens. Ex.: --max_images 100"
    )

    args = parser.parse_args()

    # ========================================================
    # CUDA
    # ========================================================
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # ========================================================
    # OUTPUT
    # ========================================================
    os.makedirs(args.output, exist_ok=True)

    watermarked_dir = os.path.join(
        args.output,
        "watermarked_images"
    )
    os.makedirs(watermarked_dir, exist_ok=True)

    # ========================================================
    # MÉTRICAS PERCEPTUAIS
    # ========================================================
    print("Carregando LPIPS...")
    lpips_metric = lpips.LPIPS(net="alex").to(device)
    lpips_metric.eval()

    print("Carregando DISTS...")
    dists_metric = DISTS().to(device)
    dists_metric.eval()

    # ========================================================
    # WATERMARK FIXA
    # ========================================================
    watermark_tensor = torch.randint(
        0,
        2,
        (1, 1, args.watermark_size, args.watermark_size),
        device=device,
        dtype=torch.float32,
    )

    watermark_path = os.path.join(
        args.output,
        "watermark_dataset.png"
    )

    watermark_np = (
        watermark_tensor[0, 0]
        .detach()
        .cpu()
        .numpy() * 255
    ).astype(np.uint8)

    cv2.imwrite(watermark_path, watermark_np)

    print(f"Watermark utilizada: {watermark_path}")

    # ========================================================
    # TRANSFORM
    # ========================================================
    image_transform = transforms.Compose([
        transforms.Resize(
            (args.image_resolution, args.image_resolution)
        ),
        transforms.ToTensor(),
    ])

    # ========================================================
    # MODELO DE EMBEDDING
    # ========================================================
    embedding_model = WatermarkEmbedding().to(device)

    checkpoint = torch.load(
        args.models_path,
        map_location=device
    )

    embedding_model.load_state_dict(
        checkpoint["embedding_model"]
    )

    # Mantém o mesmo alpha utilizado no código original.
    embedding_model.alpha = 7
    embedding_model.eval()

    print("Modelo de embedding carregado.")

    # ========================================================
    # LFW
    # ========================================================
    lfw_images = find_images(args.lfw_dir)

    if len(lfw_images) == 0:
        raise ValueError(
            f"Nenhuma imagem encontrada em: {args.lfw_dir}"
        )

    print(f"Total de imagens encontradas no LFW: {len(lfw_images)}")

    if args.max_images is not None:
        lfw_images = lfw_images[:args.max_images]

    print(
        f"Total de imagens utilizadas no teste: "
        f"{len(lfw_images)}"
    )

    # ========================================================
    # MÉTRICAS
    #
    # Todas são calculadas entre:
    #
    #       ORIGINAL  <---->  WATERMARKED
    #
    # Nenhum ataque é aplicado.
    # ========================================================
    metrics = {
        "psnr": [],
        "ssim": [],
        "lpips": [],
        "dists": [],
    }

    image_results = []

    # ========================================================
    # LOOP DAS IMAGENS
    # ========================================================
    for image_path in tqdm(lfw_images):

        # ----------------------------------------------------
        # ORIGINAL
        # ----------------------------------------------------
        image = Image.open(image_path).convert("RGB")

        image_tensor = (
            image_transform(image)
            .unsqueeze(0)
            .to(device)
        )

        # ----------------------------------------------------
        # EMBEDDING
        # ----------------------------------------------------
        with torch.no_grad():

            watermarked, _, _ = watermark_embedding_tensor(
                image_tensor=image_tensor,
                watermark_image=watermark_tensor,
                embedding_model=embedding_model,
                arnold_iterations=args.arnold_iterations,
            )

        # ----------------------------------------------------
        # SALVAR IMAGEM WATERMARKED
        # ----------------------------------------------------
        watermarked_np = (
            watermarked[0]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )

        watermarked_np = np.clip(
            watermarked_np * 255,
            0,
            255
        ).astype(np.uint8)

        # Evita colisão de nomes entre pessoas diferentes
        # do LFW que eventualmente possuam o mesmo nome de arquivo.
        person = os.path.basename(
            os.path.dirname(image_path)
        )

        filename = os.path.splitext(
            os.path.basename(image_path)
        )[0] + ".png"

        output_name = f"{person}_{filename}"

        cv2.imwrite(
            os.path.join(
                watermarked_dir,
                output_name
            ),
            cv2.cvtColor(
                watermarked_np,
                cv2.COLOR_RGB2BGR
            )
        )

        # ----------------------------------------------------
        # PSNR
        # ORIGINAL vs WATERMARKED
        # ----------------------------------------------------
        original_np = (
            image_tensor[0]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy() * 255
        ).astype(np.uint8)

        mse = np.mean(
            (
                original_np.astype(np.float32)
                - watermarked_np.astype(np.float32)
            ) ** 2
        )

        if mse == 0:
            psnr = float("inf")
        else:
            psnr = 10 * math.log10(
                (255.0 ** 2) / mse
            )

        # ----------------------------------------------------
        # SSIM
        # ORIGINAL vs WATERMARKED
        # ----------------------------------------------------
        ssim_value = structural_similarity_index_measure(
            image_tensor,
            watermarked
        ).item()

        # ----------------------------------------------------
        # LPIPS
        # ORIGINAL vs WATERMARKED
        # ----------------------------------------------------
        with torch.no_grad():

            original_lpips = (
                image_tensor * 2.0 - 1.0
            )

            watermarked_lpips = (
                watermarked * 2.0 - 1.0
            )

            lpips_value = lpips_metric(
                original_lpips,
                watermarked_lpips
            ).mean().item()

            # ------------------------------------------------
            # DISTS
            # ORIGINAL vs WATERMARKED
            # ------------------------------------------------
            dists_value = dists_metric(
                image_tensor,
                watermarked
            ).mean().item()

        # ----------------------------------------------------
        # ACUMULAR
        # ----------------------------------------------------
        metrics["psnr"].append(psnr)
        metrics["ssim"].append(ssim_value)
        metrics["lpips"].append(lpips_value)
        metrics["dists"].append(dists_value)

        image_results.append({
            "Person": person,
            "Image": os.path.basename(image_path),
            "PSNR": psnr,
            "SSIM": ssim_value,
            "LPIPS": lpips_value,
            "DISTS": dists_value,
        })

    # ========================================================
    # RESULTADO FINAL
    # ========================================================
    summary = [{
        "Dataset": "LFW",
        "Images": len(lfw_images),
        "PSNR": np.mean(metrics["psnr"]),
        "SSIM": np.mean(metrics["ssim"]),
        "LPIPS": np.mean(metrics["lpips"]),
        "DISTS": np.mean(metrics["dists"]),
    }]

    df = pd.DataFrame(summary)

    df_display = df.copy()

    df_display["PSNR"] = df_display["PSNR"].map(
        lambda x: f"{x:.2f}"
    )

    df_display["SSIM"] = df_display["SSIM"].map(
        lambda x: f"{x:.4f}"
    )

    df_display["LPIPS"] = df_display["LPIPS"].map(
        lambda x: f"{x:.4f}"
    )

    df_display["DISTS"] = df_display["DISTS"].map(
        lambda x: f"{x:.4f}"
    )

    print("\n" + "=" * 90)
    print("RESULTADOS FINAIS — LFW")
    print("Comparação: ORIGINAL vs WATERMARKED")
    print("=" * 90)

    print(
        tabulate(
            df_display,
            headers="keys",
            tablefmt="grid",
            showindex=False,
        )
    )

    # ========================================================
    # SALVAR CSV — MÉDIA
    # ========================================================
    csv_path = os.path.join(
        args.output,
        "lfw_metrics_summary.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    # ========================================================
    # SALVAR CSV — POR IMAGEM
    # ========================================================
    df_images = pd.DataFrame(image_results)

    csv_images = os.path.join(
        args.output,
        "lfw_metrics_per_image.csv"
    )

    df_images.to_csv(
        csv_images,
        index=False
    )

    print("\n===================================")
    print("Avaliação finalizada.")
    print(f"Resumo: {csv_path}")
    print(f"Por imagem: {csv_images}")
    print(f"Watermarked images: {watermarked_dir}")
    print("===================================")


if __name__ == "__main__":
    main()
