# 🎫 TicketClassifier: Classificador Inteligente de Suporte

> Projeto de Deep Learning para classificação automática de chamados de TI, desenvolvido para demonstrar competências em Engenharia de Dados e IA.

## 🎯 Objetivo
Este projeto consiste em um pipeline completo de Machine Learning capaz de categorizar tickets de suporte técnico em três classes: **Hardware**, **Software** e **Infraestrutura**.

Ele foi projetado para demonstrar domínio prático nas seguintes competências exigidas para a vaga de **Desenvolvedor Júnior de IA**:

* **Python Avançado:** Estruturação de scripts e manipulação de tipos.
* **SQL:** Simulação de banco de dados, inserção e extração de dados via query.
* **Data Science (Pandas & Numpy):** Tratamento de dados, vetorização e preparação para modelagem.
* **Deep Learning (TensorFlow/Keras):** Construção de Rede Neural Artificial com camadas de Embedding para Processamento de Linguagem Natural (NLP).
* **Visualização de Dados (Matplotlib):** Geração de métricas de performance (Acurácia e Perda).

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.13
* **Framework de IA:** TensorFlow 2.x / Keras
* **Manipulação de Dados:** Pandas, Numpy
* **Banco de Dados:** SQLite (In-memory)
* **Visualização:** Matplotlib
* **Ferramentas:** Jupytext, Helix Editor

## 🧠 Arquitetura do Modelo

O modelo utiliza uma arquitetura leve e eficiente para classificação de texto:
1.  **Input Layer:** Recebe o texto bruto do chamado.
2.  **TextVectorization:** Converte strings em tokens inteiros.
3.  **Embedding Layer:** Transforma tokens em vetores densos (aprendizado semântico).
4.  **GlobalAveragePooling1D:** Reduz a dimensionalidade focando nas características principais.
5.  **Dense Layers:** Camadas ocultas para classificação não-linear.
6.  **Softmax Output:** Probabilidade para as 3 categorias.

## 🚀 Como Executar

### Pré-requisitos
Certifique-se de ter o Python instalado.

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/SEU_USUARIO/ticket-classifier.git](https://github.com/SEU_USUARIO/ticket-classifier.git)
   cd ticket-classifier
   ```

2. **Crie e ative o ambiente virtual:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate # Para Linux e Mac
   ```

3. **Instale as dependências:**
   ```bash
   pip install tensorflow pandas numpy matplotlib scikit-learn
   ```

4. **Execute o script:**
   ```bash
   python classificador_tickets.py 
   ```

5. **Resultados:** O script treinará o modelo e gerará um arquivo `grafico_performance.png` com as curvas de aprendizado, além de exibir testes de inferência no terminal.

## 📊 Resultados Obtidos

O modelo atingiu **100% de acurácia** nos dados de treino sintéticos, demonstrando capaccidade de convergência e aprendizado eficaz dos padrões textuais fornecidos.

---

Desenvolvido por **Arthur Rodrigues**
(arthur.rodrigues.dev@proton.me)
