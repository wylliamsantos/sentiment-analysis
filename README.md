# 🚀 Análise de Sentimento em Comentários do Twitter (X)

Este projeto utiliza **Python 3.11** e a biblioteca **Hugging Face Transformers** para **analisar sentimentos em tweets**. Ele inclui **pré-processamento dos dados, treinamento de um modelo de machine learning e uma API para fazer previsões em tempo real**.

## 📌 1️⃣ **Pré-requisitos**
Antes de começar, instale os seguintes pacotes no seu sistema:
- **Python 3.11** (necessário para compatibilidade com PyTorch)
- **Pip atualizado**:
  ```bash
  pip3.11 install --upgrade pip
  ```

Verifique a versão do Python antes de continuar:
```bash
python3.11 --version
```
Saída esperada:
```
Python 3.11.x
```

---

## 📌 2️⃣ **Instalação do Ambiente Virtual**
Crie e ative um ambiente virtual para evitar conflitos de dependências:
```bash
python3.11 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

Agora, instale todas as dependências do projeto:
```bash
pip install -r requirements.txt
```

---

## 📌 3️⃣ **Execução Passo a Passo**
O fluxo de execução do projeto é o seguinte:

### 🔹 **3.1 Processamento dos Dados (`data_processing.py`)**
Este script **limpa e prepara tweets gerais do Twitter (X)** para serem usados no treinamento do modelo.

Execute:
```bash
python data_processing.py
```
📌 **O que esse script faz?**
- Remove dados irrelevantes, caracteres especiais e normaliza os textos.
- Converte os tweets limpos em um formato adequado para o modelo de machine learning.
- Salva um novo arquivo `tweets_cleaned.csv` com os dados processados.

---

### 🔹 **3.2 Verificação da Qualidade do Dataset (`check_sentiment.py`)**
Este script **verifica a distribuição dos sentimentos nos tweets processados**.

Execute:
```bash
python check_sentiment.py
```
📌 **O que esse script faz?**
- Conta quantos tweets são **Positivos, Negativos, Neutros e Irrelevantes**.
- Ajuda a decidir se devemos incluir ou excluir a classe "Irrelevante" do treinamento.

---

### 🔹 **3.3 Treinamento do Modelo (`train.py`)**
Agora, treinamos nosso modelo de aprendizado de máquina para prever sentimentos.

Execute:
```bash
python train.py
```
📌 **O que esse script faz?**
- Carrega os dados limpos (`tweets_cleaned.csv`).
- Divide os tweets em **treinamento** e **teste**.
- Treina um modelo de **BERT (bert-base-uncased)** usando o framework **Hugging Face**.
- Salva o modelo treinado em `models/sentiment_model`.

⏳ **Esse processo pode levar várias horas, dependendo do hardware**.

---

## 📌 4️⃣ **API para Predições em Tempo Real (`api.py`)**
Após treinar o modelo, podemos rodar uma **API para análise de sentimento em tempo real**.

### 🔹 **Rodando a API**
```bash
uvicorn api:app --reload
```

### 🔹 **Testando a API**
Acesse no navegador:
```
http://127.0.0.1:8000/predict/?text=I+love+this+place!
```
📌 **O que a API faz?**
- Recebe um **tweet** e retorna o sentimento (`Positive`, `Negative`, `Neutral`).
- Pode alternar entre **nosso modelo treinado** e um modelo pré-treinado (`distilbert-base-uncased-finetuned-sst-2-english`).
- O modelo a ser usado pode ser configurado na variável `USE_TRAINED_MODEL` dentro de `api.py`.

---

## 🚀 **Conclusão**
Agora você pode **rodar o projeto completo**, desde o pré-processamento até a análise de sentimento em tempo real! 🔥