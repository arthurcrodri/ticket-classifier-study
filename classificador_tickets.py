# %% [markdown]
# # Projeto: Classificador de Chamados de Suporte (TicketClassifier)
# Este script demonstra competências em:
# 1. SQL (Criação e consulta de banco de dados)
# 2. Pandas/Numpy (Manipulação de dados)
# 3. TensorFlow/Keras (Criação de Rede Neural para NLP)
# 4. Matplotlib (Visualização de métricas de treino)

# %%
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Configuração para evitar logs excessivos do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(f"Bibliotecas carregadas. TensorFlow versão: {tf.__version__}")

# %% [markdown]
# ### Simulação de Banco de Dados (SQL)
# Criamos um banco SQLite em memória e populamos com dados sintéticos.

# %%
print("\n--- PREPARANDO BANCO DE DADOS (SQL) ---")

# Conexão em memória (não cria arquivo físico)
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# Criação da tabela
cursor.execute('''
    CREATE TABLE chamados (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        descricao TEXT NOT NULL,
        categoria TEXT NOT NULL
    )
''')

# Dados de exemplo (Seed Data)
dados_base = [
    ("O servidor caiu e não consigo acessar a rede", "Infraestrutura"),
    ("A internet está muito lenta no setor financeiro", "Infraestrutura"),
    ("Não consigo logar no VPN de casa", "Infraestrutura"),
    ("Wifi não conecta no celular corporativo", "Infraestrutura"),
    ("O switch do terceiro andar está piscando vermelho", "Infraestrutura"),
    
    ("Meu mouse parou de funcionar", "Hardware"),
    ("Monitor piscando e com cores estranhas", "Hardware"),
    ("Impressora fazendo barulho e não imprime", "Hardware"),
    ("O teclado está com teclas presas", "Hardware"),
    ("A bateria do notebook não carrega", "Hardware"),
    
    ("Preciso instalar o Python e o VS Code", "Software"),
    ("Erro ao compilar o código no pipeline", "Software"),
    ("Tela azul da morte no Windows após atualização", "Software"),
    ("O Excel trava quando abro a planilha de custos", "Software"),
    ("Preciso de acesso à pasta compartilhada", "Software")
]

# Multiplicando dados para ter volume mínimo para Deep Learning (Data Augmentation simples)
dados_finais = dados_base * 40  # Total de 600 exemplos

cursor.executemany('INSERT INTO chamados (descricao, categoria) VALUES (?, ?)', dados_finais)
conn.commit()
print(f"{len(dados_finais)} chamados inseridos no banco SQL.")

# %% [markdown]
# ### Extração e Tratamento (Pandas & Numpy)

# %%
print("\n--- EXTRAÇÃO E PROCESSAMENTO (PANDAS/NUMPY) ---")

# Leitura SQL -> DataFrame Pandas
query = "SELECT descricao, categoria FROM chamados"
df = pd.read_sql(query, conn)

# Visualizando distribuição
print("Distribuição das classes:")
print(df['categoria'].value_counts())

# Transformando categorias (Texto) em Números (0, 1, 2)
le = LabelEncoder()
df['label'] = le.fit_transform(df['categoria'])
classes_nomes = le.classes_
print(f"Classes mapeadas: {classes_nomes}")

# Conversão para Numpy Arrays
X = df['descricao'].values
y = df['label'].values

# Divisão Treino (80%) e Teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Tamanho Treino: {len(X_train)} | Tamanho Teste: {len(X_test)}")

# %% [markdown]
# ### Vetorização e Modelo (TensorFlow/Deep Learning)

# %%
print("\n--- CONSTRUINDO REDE NEURAL (TENSORFLOW) ---")

# Parâmetros de NLP
VOCAB_SIZE = 1000  # Máximo de palavras no vocabulário
SEQ_LENGTH = 20    # Tamanho fixo da frase (padding)

# Camada de Vetorização (Texto -> Números Inteiros)
vectorize_layer = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQ_LENGTH
)

# Adaptar o vetorizador ao nosso texto de treino (Aprender o vocabulário)
vectorize_layer.adapt(X_train)

# Construção do Modelo Sequential
model = keras.Sequential([
    # Entrada: String bruta
    tf.keras.Input(shape=(1,), dtype=tf.string),
    
    # Vetorização
    vectorize_layer,
    
    # Embedding: Transforma índices em vetores densos (significado semântico)
    layers.Embedding(input_dim=VOCAB_SIZE + 1, output_dim=16),
    
    # Pooling: Média dos vetores para simplificar (GlobalAveragePooling1D é muito rápido)
    layers.GlobalAveragePooling1D(),
    
    # Dense: Camada oculta para aprendizado
    layers.Dense(16, activation='relu'),
    
    # Saída: 3 neurônios (softw, hardw, infra) com Softmax para probabilidade
    layers.Dense(3, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# %% [markdown]
# ### Treinamento

# %%
print("\n--- INICIANDO TREINAMENTO ---")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    verbose=1 # Mostra barra de progresso
)

# %% [markdown]
# ### Visualização (Matplotlib)

# %%
print("\n--- GERANDO GRÁFICOS (MATPLOTLIB) ---")

plt.figure(figsize=(10, 4))

# Gráfico de Acurácia
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Acurácia Treino')
plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
plt.title('Evolução da Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)

# Gráfico de Perda
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perda Treino')
plt.plot(history.history['val_loss'], label='Perda Validação')
plt.title('Evolução do Erro (Loss)')
plt.xlabel('Épocas')
plt.legend()
plt.grid(True)

nome_arquivo = 'grafico_performance.png'
plt.savefig(nome_arquivo)
print(f"Gráfico salvo como '{nome_arquivo}' na pasta atual.")

# %% [markdown]
# ### Teste Prático (Inferência)

# %%
print("\n--- TESTE DE INFERÊNCIA ---")

def classificar_texto(texto):
    # O modelo espera um Tensor
    predicoes = model.predict(tf.constant([texto]), verbose=0)
    indice_classe = np.argmax(predicoes)
    confianca = np.max(predicoes)
    nome_classe = classes_nomes[indice_classe]
    return nome_classe, confianca

# Testes manuais
frases_teste = [
    "O servidor de arquivos parou de responder",
    "Preciso que troque meu teclado, a tecla espaço quebrou",
    "Não consigo instalar o docker no linux"
]

print(f"{'TEXTO':<55} | {'PREVISÃO':<15} | {'CONFIANÇA'}")
print("-" * 85)

for frase in frases_teste:
    categoria, conf = classificar_texto(frase)
    print(f"{frase:<55} | {categoria:<15} | {conf:.1%}")

print("\nScript finalizado com sucesso!")
