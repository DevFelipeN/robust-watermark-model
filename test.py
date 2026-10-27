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
        inverse_arnold_transform
    )

    from watermark_extraction import (
        WatermarkExtraction
    )

    from attacks import AttackLayer

    import math


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

        watermark_tensor = torch.randint(
            0,
            2,
            (1, 1, args.watermark_size, args.watermark_size),
            device=device,
            dtype=torch.float32
        )

        # -------------------------
        # Transforms
        # -------------------------
        image_transform = (
            transforms.Compose([
                transforms.Resize(
                    (
                        args.image_resolution,
                        args.image_resolution
                    )
                ),
                transforms.ToTensor()
            ])
        )

        # -------------------------
        # Load model
        # -------------------------
        embedding_model = WatermarkEmbedding().to(device)
        extractor_model = WatermarkExtraction(watermark_size=args.watermark_size).to(device)

        attack_layer = AttackLayer().to(device)

        
        checkpoint = torch.load(args.models_path, map_location=device)

        embedding_model.load_state_dict(
            checkpoint["embedding_model"]
        )

        extractor_model.load_state_dict(
            checkpoint["extraction_model"]
        )

        embedding_model.alpha = 2

        embedding_model.eval()
        extractor_model.eval()
        attack_layer.eval()

        for p in embedding_model.parameters():
            print(p.mean(), p.std())
            break

        print(
            "Modelos carregados."
        )

        # -------------------------
        # Random CelebA-HQ image
        # -------------------------
        celeba_images = (
            find_images(
                args.celeba_val_dir
            )
        )

        if len(celeba_images) == 0:

            raise ValueError(
                "Nenhuma imagem encontrada "
                "na pasta de validação."
            )

        selected_image_path = (
            random.choice(
                celeba_images
            )
        )

        print(
            f"Imagem selecionada:\n"
            f"{selected_image_path}"
        )

        image = (
            Image.open(
                selected_image_path
            )
            .convert("RGB")
        )

        image_tensor = (
            image_transform(
                image
            )
            .unsqueeze(0)
            .to(device)
        )

        selected_wm = (
            watermark_tensor[0,0]
            .cpu()
            .numpy()
            * 255
        ).astype(np.uint8)

        cv2.imwrite(
            os.path.join(
                args.output,
                "selected_training_watermark.png"
            ),
            selected_wm
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
        # Diferença original x marcada
        # (calculada uma única vez, fora do loop de
        # ataques, pois não depende do ataque)
        # ==========================================
        original_np_base = (
            image_tensor[0]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
            * 255
        ).astype(np.uint8)

        watermarked_np_base = (
            watermarked[0]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )

        watermarked_np_base = np.clip(
            watermarked_np_base * 255, 0, 255
        ).astype(np.uint8)

        # -------------------------
        # Diferença REAL (abs, sem escala)
        # Fica quase toda preta, pois a marca altera
        # a imagem de forma sutil.
        # -------------------------
        diff_real = np.abs(
            original_np_base.astype(np.int16)
            - watermarked_np_base.astype(np.int16)
        )

        diff_real_img = np.clip(diff_real, 0, 255).astype(np.uint8)

        cv2.imwrite(
            os.path.join(args.output, "diferenca_real.png"),
            cv2.cvtColor(diff_real_img, cv2.COLOR_RGB2BGR)
        )

        # -------------------------
        # Diferença NORMALIZADA (com sinal, min-max)
        # -------------------------
        original_float = original_np_base.astype(np.float32) / 255.0
        watermarked_float = watermarked_np_base.astype(np.float32) / 255.0

        residual = watermarked_float - original_float

        r_min, r_max = residual.min(), residual.max()

        if r_max - r_min > 1e-8:
            diff_normalizada = (residual - r_min) / (r_max - r_min)
        else:
            diff_normalizada = np.zeros_like(residual)

        diff_normalizada_img = (diff_normalizada * 255).astype(np.uint8)

        cv2.imwrite(
            os.path.join(args.output, "diferenca_normalizada.png"),
            cv2.cvtColor(diff_normalizada_img, cv2.COLOR_RGB2BGR)
        )

        print(
            "\n========== DIFERENÇA ORIGINAL x MARCADA ==========\n"
            f"Diferença real (abs) - média: {diff_real.mean():.4f} | máxima: {diff_real.max():.4f}\n"
            f"Residual (com sinal) - mínimo: {r_min:.4f} | máximo: {r_max:.4f}\n"
            "===================================================\n"
        )

        attacks = [
            "none",
            "jpeg",
            "blur",
            "median_blur",
            "noise",
            "salt_pepper",
            "dropout",
            "resize"
        ]

        for attack in attacks:

            print(f"\n========== {attack.upper()} ==========")
            attack_dir = os.path.join(args.output, attack)
            os.makedirs(attack_dir, exist_ok=True)

            attacked, attack_info = attack_layer(
                watermarked,
                attack=attack
            )

            print(attack_info)

            # -------------------------
            # Prediction
            # -------------------------
            with torch.no_grad():

                pred = extractor_model(attacked)

                pred = (
                    torch.sigmoid(
                        pred
                    ) > 0.5
                ).float()

                # Marca inserida (target)
                target_img = (
                    watermark_tensor[0,0]
                    .detach()
                    .cpu()
                    .numpy()
                    * 255
                ).astype(np.uint8)

                # Marca extraída (pred)
                pred_img = (
                    pred[0, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    * 255
                ).astype(np.uint8)

                cv2.imwrite(
                    os.path.join(
                        attack_dir,
                        "target_raw.png"
                    ),
                    target_img
                )

                cv2.imwrite(
                    os.path.join(
                        attack_dir,
                        "pred_raw.png"
                    ),
                    pred_img
                )


            # -------------------------
            # inverse arnold
            # -------------------------
            watermark_original = (
                inverse_arnold_transform(
                    watermark_tensor[0,0],
                    args.arnold_iterations
                )
            )

            watermark_extracted = (
                inverse_arnold_transform(
                    pred[0,0],
                    args.arnold_iterations
                )
            )

            # -------------------------
            # tensor -> numpy
            # -------------------------
            watermark_original = (
                watermark_original
                .cpu()
                .numpy()
                * 255
            ).astype(np.uint8)

            watermark_extracted = (
                watermark_extracted
                .cpu()
                .numpy()
                * 255
            ).astype(np.uint8)

            # -------------------------
            # BER / Bit Accuracy
            # -------------------------
            watermark_original_binary = (
                watermark_original > 127
            ).astype(
                np.uint8
            )

            watermark_extracted_binary = (
                watermark_extracted > 127
            ).astype(
                np.uint8
            )

            bit_errors = np.sum(
                watermark_original_binary
                !=
                watermark_extracted_binary
            )

            total_bits = (
                watermark_original_binary
                .size
            )

            ber = (
                bit_errors
                / total_bits
            )

            bit_acc = (
                1.0 - ber
            )

            accuracy = (
                bit_acc * 100.0
            )

            # -------------------------
            # Prompt output
            # -------------------------
            print(
                "\n========== MÉTRICAS =========="
            )

            print(
                f"BER: {ber:.4f}"
            )

            print(
                f"Bit Accuracy: "
                f"{bit_acc:.4f}"
            )

            print(
                f"Accuracy: "
                f"{accuracy:.2f}%"
            )

            print(
                "==============================\n"
            )
            

            # -------------------------
            # save images
            # -------------------------
            original_np = (
                image_tensor[0]
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
                * 255
            ).astype(np.uint8)

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
            ).astype(np.uint8)

            watermarked_bgr = (
                cv2.cvtColor(
                    watermarked_np,
                    cv2.COLOR_RGB2BGR
                )
            )

            original_bgr = cv2.cvtColor(
                original_np,
                cv2.COLOR_RGB2BGR
            )
            
            attacked_np = (
                attacked[0]
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy() * 255
            ).astype(np.uint8)

            cv2.imwrite(
                os.path.join(
                    attack_dir,
                    "attacked.png"
                ),
                cv2.cvtColor(attacked_np, cv2.COLOR_RGB2BGR)
            )

            cv2.imwrite(
                os.path.join(
                    attack_dir,
                    "original.png"
                ),
                original_bgr
            )

            cv2.imwrite(
                os.path.join(
                    attack_dir,
                    "watermarked.png"
                ),
                watermarked_bgr
            )

            cv2.imwrite(
                os.path.join(
                    attack_dir,
                    "watermark_original.png"
                ),
                watermark_original
            )

            cv2.imwrite(
                os.path.join(
                    attack_dir,
                    "watermark_extracted.png"
                ),
                watermark_extracted
            )

            print(
                "Resultados salvos."
            )


            # -------------------------
            # PSNR
            # -------------------------
            mse = np.mean(
                (
                    original_np.astype(np.float32)
                    -
                    attacked_np.astype(np.float32)
                ) ** 2
            )

            if mse == 0:
                psnr = float("inf")
            else:
                psnr = 10 * math.log10((255.0 ** 2) / mse)

            # -------------------------
            # show grid
            # -------------------------
            # Só na iteração "none" (sem ataque) mostramos os
            # painéis extras de diferença (real e normalizada),
            # já que é aqui que a comparação original x marcada
            # faz sentido. Nos demais ataques, mantém-se o grid
            # original de 4 painéis.
            # -------------------------
            if attack == "none":

                fig, axes = plt.subplots(
                    1,
                    6,
                    figsize=(26,5)
                )

                axes[0].imshow(original_np)
                axes[0].set_title("Imagem Original")
                axes[0].axis("off")

                axes[1].imshow(diff_real_img)
                axes[1].set_title("Diferença Real (abs)")
                axes[1].axis("off")

                axes[2].imshow(diff_normalizada_img)
                axes[2].set_title("Diferença Normalizada")
                axes[2].axis("off")

                axes[3].imshow(attacked_np)
                axes[3].set_title(f"Attack: {attack}")
                axes[3].axis("off")

                axes[4].imshow(
                    watermark_original,
                    cmap="gray"
                )
                axes[4].set_title("Marca Original")
                axes[4].axis("off")

                axes[5].imshow(
                    watermark_extracted,
                    cmap="gray"
                )
                axes[5].set_title("Marca Extraída")
                axes[5].axis("off")

            else:

                fig, axes = plt.subplots(
                    1,
                    4,
                    figsize=(18,5)
                )

                # -------------------------
                # Original
                # -------------------------
                axes[0].imshow(original_np)
                axes[0].set_title("Imagem Original")
                axes[0].axis("off")

                # -------------------------
                # Marcada + ataque
                # -------------------------
                axes[1].imshow(attacked_np)
                axes[1].set_title(f"Attack: {attack}")
                axes[1].axis("off")

                # -------------------------
                # Marca original
                # -------------------------
                axes[2].imshow(
                    watermark_original,
                    cmap="gray"
                )
                axes[2].set_title("Marca Original")
                axes[2].axis("off")

                # -------------------------
                # Marca extraída
                # -------------------------
                axes[3].imshow(
                    watermark_extracted,
                    cmap="gray"
                )
                axes[3].set_title("Marca Extraída")
                axes[3].axis("off")

            # -------------------------
            # Métricas
            # -------------------------
            

            titulo = (
                f"PSNR = {psnr:.2f} dB | "
                f"BitAcc = {bit_acc:.4f} | "
                f"Accuracy = {accuracy:.2f}% | "
                f"BER = {ber:.4f}"
            )

            fig.suptitle(
                titulo,
                fontsize=14,
                fontweight="bold"
            )

            plt.tight_layout(rect=[0,0,1,0.92])

            plt.savefig(
                os.path.join(
                    attack_dir,
                    "resultado.png"
                ),
                dpi=300
            )

            plt.show()

        '''
        mod = modification[0].detach().cpu().numpy()

        band_names = [
            "cH_R",
            "cV_R",
            "cH_G",
            "cV_G",
            "cH_B",
            "cV_B"
        ]

        for i, name in enumerate(band_names):

            m = mod[i]

            # normaliza apenas para visualização
            m = (m - m.min()) / (m.max() - m.min() + 1e-8)

            m = (255 * m).astype(np.uint8)

            cv2.imwrite(
                os.path.join(
                    args.output,
                    f"modification_{name}.png"
                ),
                m
            )

        print("min :", modification.min().item())
        print("max :", modification.max().item())
        print("mean:", modification.mean().item())
        print("std :", modification.std().item())
        '''

    if __name__ == "__main__":
        main()