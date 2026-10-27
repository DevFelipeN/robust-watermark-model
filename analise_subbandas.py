import os
import argparse
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from watermark_embedding import (
    WatermarkEmbedding,
    watermark_embedding_tensor,
    block_dct2d,
    dwt2_torch
)

from pytorch_wavelets import DWTForward


# ==========================================
# FIND IMAGES
# ==========================================
def find_images(root):

    paths = []

    for dirpath, _, filenames in os.walk(root):

        for f in filenames:

            if f.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            ):

                paths.append(
                    os.path.join(
                        dirpath,
                        f
                    )
                )

    return sorted(paths)


def block_dct2d_rgb(image):
    B, C, H, W = image.shape

    image_2d = image.reshape(
        B * C,
        H,
        W
    )

    dct_2d = block_dct2d(
        image_2d,
        block_size=8
    )

    return dct_2d.reshape(
        B,
        C,
        H,
        W
    )


def transform_dct_dwt(
    image_tensor,
    dwt
):

    # ==================================================
    # DCT
    # ==================================================

    image = image_tensor * 255.0

    dct_image = block_dct2d_rgb(
        image
    )

    # ==================================================
    # DWT
    # ==================================================

    subbandas = dwt(
        dct_image
    )

    return subbandas


# ==========================================
# MAIN
# ==========================================
def main():

    parser = argparse.ArgumentParser()

    # -------------------------
    # Arguments
    # -------------------------

    parser.add_argument(
        "--models_path",
        required=True
    )

    parser.add_argument(
        "--celeba_val_dir",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--image_resolution",
        type=int,
        default=128
    )

    parser.add_argument(
        "--watermark_size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--arnold_iterations",
        type=int,
        default=0
    )

    parser.add_argument(
        "--cuda",
        type=int,
        default=0
    )

    args = parser.parse_args()

    # -------------------------
    # CUDA
    # -------------------------

    os.environ[
        "CUDA_VISIBLE_DEVICES"
    ] = str(
        args.cuda
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # -------------------------
    # Output
    # -------------------------

    os.makedirs(
        args.output,
        exist_ok=True
    )

    # -------------------------
    # Watermark aleatória
    # -------------------------

    watermark_tensor = torch.randint(
        0,
        2,
        (
            1,
            1,
            args.watermark_size,
            args.watermark_size
        ),
        device=device,
        dtype=torch.float32
    )

    # -------------------------
    # Transform
    # -------------------------

    image_transform = transforms.Compose(
        [
            transforms.Resize(
                (
                    args.image_resolution,
                    args.image_resolution
                )
            ),
            transforms.ToTensor()
        ]
    )

    # -------------------------
    # Load model
    # -------------------------

    embedding_model = (
        WatermarkEmbedding()
        .to(device)
    )

    checkpoint = torch.load(
        args.models_path,
        map_location=device
    )

    embedding_model.load_state_dict(
        checkpoint["embedding_model"]
    )

    # Mesmo alpha utilizado no código
    # de referência
    embedding_model.alpha = 5

    embedding_model.eval()

    print(
        "Modelo de embedding carregado."
    )

    # -------------------------
    # Find images
    # -------------------------

    celeba_images = find_images(
        args.celeba_val_dir
    )

    if len(celeba_images) == 0:

        raise ValueError(
            "Nenhuma imagem encontrada "
            "na pasta de validação."
        )

    # -------------------------
    # Selecionar imagem
    # -------------------------

    selected_image_path = random.choice(
        celeba_images
    )

    print(
        f"\nImagem selecionada:\n"
        f"{selected_image_path}"
    )

    # -------------------------
    # Load image
    # -------------------------

    image = (
        Image.open(
            selected_image_path
        )
        .convert("RGB")
    )

    # -------------------------
    # Resize para 128x128
    # -------------------------

    image_tensor = (
        image_transform(image)
        .unsqueeze(0)
        .to(device)
    )

    print(
        f"Dimensão da imagem: "
        f"{image_tensor.shape}"
    )

    # ==================================================
    # EMBEDDING
    # ==================================================

    with torch.no_grad():

        watermarked, target, modification = (
            watermark_embedding_tensor(
                image_tensor=image_tensor,
                watermark_image=watermark_tensor,
                embedding_model=embedding_model,
                arnold_iterations=args.arnold_iterations
            )
        )

    print(
        f"Imagem marcada: "
        f"{watermarked.shape}"
    )

    # ==================================================
    # SALVAR ORIGINAL
    # ==================================================

    original_np = (
        image_tensor[0]
        .permute(
            1,
            2,
            0
        )
        .detach()
        .cpu()
        .numpy()
        * 255
    )

    original_np = np.clip(
        original_np,
        0,
        255
    ).astype(
        np.uint8
    )

    cv2.imwrite(
        os.path.join(
            args.output,
            "original_128x128.png"
        ),
        cv2.cvtColor(
            original_np,
            cv2.COLOR_RGB2BGR
        )
    )

    # ==================================================
    # SALVAR MARCADA
    # ==================================================

    watermarked_np = (
        watermarked[0]
        .permute(
            1,
            2,
            0
        )
        .detach()
        .cpu()
        .numpy()
        * 255
    )

    watermarked_np = np.clip(
        watermarked_np,
        0,
        255
    ).astype(
        np.uint8
    )

    cv2.imwrite(
        os.path.join(
            args.output,
            "watermarked_128x128.png"
        ),
        cv2.cvtColor(
            watermarked_np,
            cv2.COLOR_RGB2BGR
        )
    )

    # ==================================================
    # DIFERENÇA A NÍVEL DE PIXEL (SEM DCT/DWT)
    # ==================================================
    #
    # Diferença absoluta direta entre a imagem original e
    # a marcada, no domínio do pixel (0-255), SEM aplicar
    # DCT ou DWT. Serve como referência "crua" para
    # comparar com as diferenças nas subbandas.
    #
    # Calculada a partir dos tensores em ponto flutuante
    # (image_tensor, watermarked), antes de qualquer
    # arredondamento/clip para uint8, para manter a maior
    # precisão possível.
    #
    # ==================================================

    diff_pixel = torch.abs(
        (watermarked * 255.0)
        -
        (image_tensor * 255.0)
    )

    print(
        "\ndiff_pixel:",
        diff_pixel.shape
    )

    # -------------------------
    # RGB -> numpy (H, W, 3)
    # -------------------------

    diff_pixel_rgb = (
        diff_pixel[0]
        .detach()
        .cpu()
        .numpy()
        .transpose(
            1,
            2,
            0
        )
    )

    # -------------------------
    # Estatísticas
    # -------------------------

    print(
        f"Pixel -> média: "
        f"{diff_pixel_rgb.mean():.8f} | "
        f"máximo: "
        f"{diff_pixel_rgb.max():.8f}"
    )

    # -------------------------
    # Magnitude (RGB -> 2D)
    # -------------------------

    diff_pixel_mag = np.sqrt(
        np.sum(diff_pixel_rgb ** 2, axis=2)
    )

    pixel_max = diff_pixel_mag.max()

    # -------------------------
    # Salvar array bruto (.npy)
    # -------------------------

    np.save(
        os.path.join(
            args.output,
            "diff_pixel.npy"
        ),
        diff_pixel_mag
    )

    # -------------------------
    # Salvar imagem RGB da diferença (sem normalizar
    # pela escala global das subbandas, já que está em
    # outra unidade/domínio — usa o próprio máximo da
    # diferença de pixel como referência)
    # -------------------------

    if pixel_max > 0:

        diff_pixel_rgb_norm = (
            diff_pixel_rgb
            /
            pixel_max
        )

    else:

        diff_pixel_rgb_norm = (
            diff_pixel_rgb
            *
            0.0
        )

    diff_pixel_rgb_uint8 = np.clip(
        diff_pixel_rgb_norm * 255.0,
        0,
        255
    ).astype(
        np.uint8
    )

    cv2.imwrite(
        os.path.join(
            args.output,
            "diff_pixel_rgb.png"
        ),
        cv2.cvtColor(
            diff_pixel_rgb_uint8,
            cv2.COLOR_RGB2BGR
        )
    )

    # -------------------------
    # Salvar imagem em escala de cinza (magnitude)
    # -------------------------

    if pixel_max > 0:

        diff_pixel_mag_norm = (
            diff_pixel_mag
            /
            pixel_max
        )

    else:

        diff_pixel_mag_norm = (
            diff_pixel_mag
            *
            0.0
        )

    diff_pixel_mag_uint8 = np.clip(
        diff_pixel_mag_norm * 255.0,
        0,
        255
    ).astype(
        np.uint8
    )

    cv2.imwrite(
        os.path.join(
            args.output,
            "diff_pixel_gray.png"
        ),
        diff_pixel_mag_uint8
    )

    print(
        "Diferença a nível de pixel salva em:"
    )

    print(
        f"  {os.path.join(args.output, 'diff_pixel.npy')}"
    )

    print(
        f"  {os.path.join(args.output, 'diff_pixel_rgb.png')}"
    )

    print(
        f"  {os.path.join(args.output, 'diff_pixel_gray.png')}"
    )

    # ==================================================
    # DWT
    # ==================================================

    dwt = DWTForward(
        J=1,
        wave="haar"
    ).to(device)

    # ==================================================
    # TRANSFORMAÇÃO DA IMAGEM ORIGINAL
    # ==================================================

    print(
        "\nAplicando DCT + DWT na imagem original..."
    )

    original_subbands = transform_dct_dwt(
        image_tensor,
        dwt
    )

    # ==================================================
    # TRANSFORMAÇÃO DA IMAGEM MARCADA
    # ==================================================

    print(
        "Aplicando DCT + DWT na imagem marcada..."
    )

    watermarked_subbands = transform_dct_dwt(
        watermarked,
        dwt
    )

    # ==================================================
    # DIFERENÇA DAS SUB-BANDAS
    # ==================================================

    original_LL, original_high = (
        original_subbands
    )

    watermarked_LL, watermarked_high = (
        watermarked_subbands
    )

    original_LH = (
        original_high[0][
            :, :, 0, :, :
        ]
    )

    original_HL = (
        original_high[0][
            :, :, 1, :, :
        ]
    )

    original_HH = (
        original_high[0][
            :, :, 2, :, :
        ]
    )

    watermarked_LH = (
        watermarked_high[0][
            :, :, 0, :, :
        ]
    )

    watermarked_HL = (
        watermarked_high[0][
            :, :, 1, :, :
        ]
    )

    watermarked_HH = (
        watermarked_high[0][
            :, :, 2, :, :
        ]
    )

    # ==================================================
    # DIFERENÇA ABSOLUTA
    # ==================================================

    diff_LL = torch.abs(
        watermarked_LL
        -
        original_LL
    )

    diff_LH = torch.abs(
        watermarked_LH
        -
        original_LH
    )

    diff_HL = torch.abs(
        watermarked_HL
        -
        original_HL
    )

    diff_HH = torch.abs(
        watermarked_HH
        -
        original_HH
    )

    print(
        "\nSHAPES DAS DIFERENÇAS:"
    )

    print(
        "diff_LL:",
        diff_LL.shape
    )

    print(
        "diff_LH:",
        diff_LH.shape
    )

    print(
        "diff_HL:",
        diff_HL.shape
    )

    print(
        "diff_HH:",
        diff_HH.shape
    )

    # ==================================================
    # REMOVER DIMENSÕES DESNECESSÁRIAS
    # ==================================================

    diff_LL = (
        diff_LL[0]
        .detach()
        .cpu()
        .numpy()
        .transpose(
            1,
            2,
            0
        )
    )

    diff_LH = (
        diff_LH[0]
        .detach()
        .cpu()
        .numpy()
        .transpose(
            1,
            2,
            0
        )
    )

    diff_HL = (
        diff_HL[0]
        .detach()
        .cpu()
        .numpy()
        .transpose(
            1,
            2,
            0
        )
    )

    diff_HH = (
        diff_HH[0]
        .detach()
        .cpu()
        .numpy()
        .transpose(
            1,
            2,
            0
        )
    )

    # ==================================================
    # ESTATÍSTICAS
    # ==================================================

    print(
        "\n=============================================="
    )

    print(
        "DIFERENÇA ENTRE ORIGINAL E MARCADA"
    )

    print(
        "=============================================="
    )

    print(
        f"LL -> média: "
        f"{diff_LL.mean():.8f} | "
        f"máximo: "
        f"{diff_LL.max():.8f}"
    )

    print(
        f"LH -> média: "
        f"{diff_LH.mean():.8f} | "
        f"máximo: "
        f"{diff_LH.max():.8f}"
    )

    print(
        f"HL -> média: "
        f"{diff_HL.mean():.8f} | "
        f"máximo: "
        f"{diff_HL.max():.8f}"
    )

    print(
        f"HH -> média: "
        f"{diff_HH.mean():.8f} | "
        f"máximo: "
        f"{diff_HH.max():.8f}"
    )

    print(
        "=============================================="
    )

    # ==================================================
    # ESCALA GLOBAL PARA VISUALIZAÇÃO
    # ==================================================
    #
    # IMPORTANTE:
    #
    # Não normalizamos cada subbanda separadamente.
    #
    # Todas as quatro imagens usam a mesma escala.
    #
    # Assim podemos comparar visualmente:
    #
    #     LL
    #     LH
    #     HL
    #     HH
    #
    # usando exatamente a mesma referência de intensidade.
    #
    # ==================================================

    # ==================================================
    # CONVERTER RGB -> MAGNITUDE
    # ==================================================

    diff_LL_mag = np.sqrt(
        np.sum(diff_LL ** 2, axis=2)
    )

    diff_LH_mag = np.sqrt(
        np.sum(diff_LH ** 2, axis=2)
    )

    diff_HL_mag = np.sqrt(
        np.sum(diff_HL ** 2, axis=2)
    )

    diff_HH_mag = np.sqrt(
        np.sum(diff_HH ** 2, axis=2)
    )

    # ==================================================
    # ESCALA GLOBAL
    # ==================================================

    global_max = max(
        diff_LL_mag.max(),
        diff_LH_mag.max(),
        diff_HL_mag.max(),
        diff_HH_mag.max()
    )

    print(
        f"\nEscala global da visualização: "
        f"0.0 -> {global_max:.8f}"
    )

    # ==================================================
    # SALVAR DIFERENÇA ABSOLUTA DE CADA SUBBANDA
    # ==================================================
    #
    # Salva tanto os arrays brutos (.npy), preservando os
    # valores numéricos reais das diferenças para análise
    # posterior, quanto imagens em escala de cinza (.png),
    # normalizadas pela MESMA escala global (global_max)
    # usada no plot comparativo — assim as 4 imagens salvas
    # permanecem comparáveis entre si em intensidade.
    #
    # ==================================================

    subbands_mag = {
        "LL": diff_LL_mag,
        "LH": diff_LH_mag,
        "HL": diff_HL_mag,
        "HH": diff_HH_mag
    }

    print(
        "\nSalvando diferenças absolutas de cada subbanda..."
    )

    for nome, mag in subbands_mag.items():

        # -------------------------
        # Array bruto (.npy)
        # -------------------------

        npy_path = os.path.join(
            args.output,
            f"diff_{nome}.npy"
        )

        np.save(
            npy_path,
            mag
        )

        # -------------------------
        # Imagem normalizada (.png)
        # -------------------------
        #
        # Normalização pela escala GLOBAL (global_max),
        # não pelo máximo individual de cada subbanda,
        # para manter a mesma referência de intensidade
        # usada no gráfico comparativo.
        #
        # -------------------------

        if global_max > 0:

            mag_norm = (
                mag / global_max
            )

        else:

            mag_norm = (
                mag * 0.0
            )

        mag_uint8 = np.clip(
            mag_norm * 255.0,
            0,
            255
        ).astype(
            np.uint8
        )

        png_path = os.path.join(
            args.output,
            f"diff_{nome}.png"
        )

        cv2.imwrite(
            png_path,
            mag_uint8
        )

        print(
            f"  {nome} -> "
            f"{npy_path} | "
            f"{png_path}"
        )

    # ==================================================
    # MOSTRAR AS 4 SUB-BANDAS DA DIFERENÇA
    # ==================================================
    #
    # CORREÇÃO: usamos as versões de MAGNITUDE (2D)
    # das diferenças, e não os arrays RGB originais.
    #
    # O imshow ignora silenciosamente cmap/vmin/vmax
    # quando recebe um array (H, W, 3) — por isso a
    # escala global calculada acima não tinha efeito
    # nenhum na versão anterior do código.
    #
    # ==================================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            12,
            10
        )
    )

    # -------------------------
    # LL
    # -------------------------

    axes[0, 0].imshow(
        diff_LL_mag,
        cmap="gray",
        vmin=0,
        vmax=global_max
    )

    axes[0, 0].set_title(
        "Diferença — LL"
    )

    axes[0, 0].axis(
        "off"
    )

    # -------------------------
    # LH
    # -------------------------

    axes[0, 1].imshow(
        diff_LH_mag,
        cmap="gray",
        vmin=0,
        vmax=global_max
    )

    axes[0, 1].set_title(
        "Diferença — LH"
    )

    axes[0, 1].axis(
        "off"
    )

    # -------------------------
    # HL
    # -------------------------

    axes[1, 0].imshow(
        diff_HL_mag,
        cmap="gray",
        vmin=0,
        vmax=global_max
    )

    axes[1, 0].set_title(
        "Diferença — HL"
    )

    axes[1, 0].axis(
        "off"
    )

    # -------------------------
    # HH
    # -------------------------

    axes[1, 1].imshow(
        diff_HH_mag,
        cmap="gray",
        vmin=0,
        vmax=global_max
    )

    axes[1, 1].set_title(
        "Diferença — HH"
    )

    axes[1, 1].axis(
        "off"
    )

    fig.suptitle(
        "Diferença DCT + DWT — Original × Imagem Marcada",
        fontsize=15,
        fontweight="bold"
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.95
        ]
    )

    # ==================================================
    # SALVAR FIGURA
    # ==================================================

    output_path = os.path.join(
        args.output,
        "diferenca_dct_dwt.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"\nFigura salva em:\n"
        f"{output_path}"
    )

    # ==================================================
    # MOSTRAR NA TELA
    # ==================================================

    plt.show()


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    main()