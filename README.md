# Experimento classificação de imagens com modelos de aprendizado de máquina

## Executando no kaggle

1. Acesse URL
2. Faça uma cópia do notebook
3. Ajuste a aceleração para P100
4. Execute todas as células

## Executando local

0. Pré-requisitos
1. Criar ambiente
2. Executar ambiente
3. Instalar dependências
4. Executar

### 0. Pré-requisitos

1. Conda instalado
2. Jupyter notebook
3. Hardware mínimo
- Os requisitos de hardware para alguns modelos são no mínimo:

### 1. Criar o ambiente

```bash
conda create -n ml-exp python=3.7
```

### 2. Executar ambiente

```bash
conda activate ml-exp
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt -r requirements-local.txt
```

### 4. Executar

Executando em linux, mac os ou windows-linux wsl2:

```bash
jupyter-lab
```

> Caso queira coletar métricas de execução, configure uma variável de ambiente com a chave obtida no site [wandb.ai](https://docs.wandb.ai/quickstart).