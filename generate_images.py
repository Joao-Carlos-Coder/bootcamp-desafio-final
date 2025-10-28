import torch
import os
import argparse
import matplotlib.pyplot as plt

# Importa as classes do modelo do nosso script de treinamento original
from training_script import CVAE, one_hot

def generate_images(args):
    """
    Carrega o modelo CVAE treinado e gera uma imagem para cada dígito (0-9).
    """
    # Define o dispositivo (CPU, pois o GitHub Actions não tem GPU)
    device = torch.device('cpu')
    logger.info("Usando dispositivo: %s", device)

    # Inicializa a arquitetura do modelo com os mesmos parâmetros do treinamento
    model = CVAE(
        input_channels=1,
        num_classes=args.num_classes,
        latent_dim=args.latent_dim
    ).to(device)
    logger.info("Arquitetura do modelo inicializada.")

    # Carrega os "pesos" treinados do arquivo .pth
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        logger.info("Modelo treinado carregado com sucesso de: %s", args.model_path)
    except FileNotFoundError:
        logger.error("Arquivo do modelo não encontrado em: %s. Verifique o caminho.", args.model_path)
        return
    except Exception as e:
        logger.error("Erro ao carregar o modelo: %s", e)
        return

    # Coloca o modelo em modo de "avaliação" (importante para a geração)
    model.eval()

    logger.info("Gerando uma imagem para cada dígito de 0 a 9...")
    
    # Cria uma grade de 2x5 para plotar todas as 10 imagens
    fig, axs = plt.subplots(2, 5, figsize=(10, 5))
    fig.suptitle('Dígitos Gerados pelo Modelo CVAE', fontsize=16)

    # Usa o modo `no_grad` para economizar memória e computação
    with torch.no_grad():
        for i in range(args.num_classes):
            # 1. Cria um ponto aleatório no espaço latente (a "inspiração" para a pintura)
            z = torch.randn(1, args.latent_dim).to(device)

            # 2. Especifica qual dígito queremos (o "tema" da pintura)
            label = torch.tensor([i]).to(device)
            label_one_hot = one_hot(label, args.num_classes, device)

            # 3. Pede ao Decoder para gerar a imagem
            generated_image_tensor = model.decoder(z, label_one_hot)

            # 4. Converte o tensor para um formato que possa ser exibido e salvo
            image = generated_image_tensor.squeeze().cpu().numpy()
            
            # 5. Plota a imagem na sua posição correta na grade
            row, col = i // 5, i % 5
            ax = axs[row, col]
            ax.imshow(image, cmap='gray')
            ax.set_title(f'Gerado: {i}')
            ax.axis('off')

    # Salva a grade completa de imagens em um arquivo
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.output_path)
    logger.info("Grade de imagens salva em: %s", args.output_path)


if __name__ == '__main__':
    # Configura um logger básico para vermos as mensagens no log do GitHub Actions
    import logging
    import sys
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    
    parser = argparse.ArgumentParser(description="Gera imagens usando um modelo CVAE treinado.")
    
    # Argumentos para localizar o modelo e salvar a saída
    parser.add_argument("--model-path", type=str, required=True, help="Caminho para o arquivo do modelo treinado (model.pth).")
    parser.add_argument("--output-path", type=str, default="generated_digits.png", help="Caminho para salvar a imagem de saída.")
    
    # Argumentos que definem a arquitetura do modelo (devem ser os mesmos do treino)
    parser.add_argument("--latent_dim", type=int, default=120)
    parser.add_argument("--num-classes", type=int, default=10)

    args = parser.parse_args()
    generate_images(args)
