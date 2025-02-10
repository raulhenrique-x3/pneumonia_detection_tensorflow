# Detecção de Pneumonia em Imagens de Raio-X com Deep Learning

Este repositório contém um projeto de aprendizado profundo para a detecção de pneumonia em imagens de raio-X utilizando diferentes modelos de redes neurais convolucionais. O objetivo é comparar o desempenho de uma CNN personalizada e modelos pré-treinados (VGG16 e ResNet50).

## Estrutura do Projeto

```plaintext
DEEPLEARNING_PROJECT/
│
├── data/                    # Dados de treino e teste
│   ├── archive/
│       ├── chest_xray/
│           ├── train/
│           ├── test/
├── models/                  # Modelos treinados salvos
├── pneumonia-env/           # Ambiente virtual do projeto
├── .gitignore               # Arquivos ignorados pelo Git
├── main.py                  # Script principal do projeto
├── README.md                # Documentação do projeto
├── requirements.txt         # Bibliotecas necessárias
```

## Requisitos do Sistema

- Python 3.8+
- TensorFlow 2.0+
- GPU com suporte a CUDA (opcional, recomendado para desempenho)

## Instalação

1. Clone o repositório:

   ```bash
   git clone https://github.com/seu_usuario/DEEPLEARNING_PROJECT.git
   cd DEEPLEARNING_PROJECT
   ```

2. Crie um ambiente virtual e ative-o:

   ```bash
   python -m venv pneumonia-env
   source pneumonia-env/bin/activate   # No Windows: pneumonia-env\Scripts\activate
   ```

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure os dados:
   - Baixe o dataset de raio-X.
   - Estruture os dados conforme o diretório `data/archive/chest_xray`.

## Uso

1. Execute o script principal para treinar e avaliar os modelos:

   ```bash
   python main.py
   ```

2. Os modelos treinados serão salvos na pasta `models/`.

## Modelos Implementados

### 1. CNN Personalizada

Uma arquitetura simples com:

- 3 camadas convolucionais
- Max pooling
- Dropout para regularização

### 2. VGG16

Modelo pré-treinado com ajuste fino das últimas camadas.

### 3. ResNet50

Modelo pré-treinado com aprendizado residual e ajuste fino.

## Resultados

Os modelos foram avaliados no conjunto de teste:

| Modelo            | Acurácia no Teste |
| ----------------- | ----------------- |
| CNN Personalizada | 91,03%            |
| VGG16             | 86,38%            |
| ResNet50          | 92,63%            |

## Métricas

- Acurácia
- Perda de validação e treinamento
- Gráficos de desempenho gerados automaticamente no treinamento.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir uma issue ou enviar um pull request.

---
