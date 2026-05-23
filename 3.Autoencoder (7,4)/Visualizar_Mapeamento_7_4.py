import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from libs.val_model import recover_weights
from libs.topology import Net_Coded
from libs.model_E2E import End2EndSystem
from scipy.spatial.distance import cdist

k = 4
n = 7
M = 2**k

WEIGHTS_DIR_BMI = "./Modelos/Pesos/FullyConnected/BMI"
WEIGHTS_DIR_MI = "./Modelos/Pesos/FullyConnected/MI"

bmi_tx = Net_Coded.encoder(k, n)
bmi_rx = Net_Coded.decoder(k, n, a=4, bmi=True)
model_bmi = End2EndSystem(k, n, bmi_tx, bmi_rx, training=False, bit_wise=True)
_ = model_bmi(1, 10.0)
model_bmi = recover_weights(model_bmi, f"{WEIGHTS_DIR_BMI}/weights-BMI-k4-n7-a4")

mi_tx = Net_Coded.encoder(k, n)
mi_rx = Net_Coded.decoder(k, n, a=2, bmi=False)
model_mi = End2EndSystem(k, n, mi_tx, mi_rx, training=False, bit_wise=False)
_ = model_mi(1, 10.0)
model_mi = recover_weights(model_mi, f"{WEIGHTS_DIR_MI}/weights-MI-k4-n7-a2")

inputs = tf.eye(M)
symbols_bmi = model_bmi.transmitter(inputs).numpy()
symbols_mi = model_mi.transmitter(inputs).numpy()

# Ordenando as sequências binárias usando Gray Code
gray_indices = [0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8]
labels = [f"{i:04b}" for i in gray_indices]

# Reordenando os símbolos extraídos
symbols_bmi = symbols_bmi[gray_indices]
symbols_mi = symbols_mi[gray_indices]

dist_bmi = cdist(symbols_bmi, symbols_bmi, metric='euclidean')
dist_mi = cdist(symbols_mi, symbols_mi, metric='euclidean')

out_dir = "./Figures/Fully Connected/3.Autoencoder (7,4)"
os.makedirs(out_dir, exist_ok=True)

# -------------------------------------------------------------------------------------------------
# 1. Matriz de Distâncias Euclidianas com Seaborn
# -------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Matriz de Distâncias Euclidianas: BMI vs MI (Autoencoder 7,4)', fontsize=16, fontweight='bold')

mask = np.eye(M, dtype=bool)

sns.heatmap(dist_bmi, annot=True, fmt=".1f", cmap="rocket_r", mask=mask, linewidths=0.5, linecolor='gray', ax=axes[0],
            xticklabels=labels, yticklabels=labels, cbar_kws={'fraction': 0.046, 'pad': 0.04})
axes[0].set_title('BMI (Bit-wise, a=4)')

sns.heatmap(dist_mi, annot=True, fmt=".1f", cmap="rocket_r", mask=mask, linewidths=0.5, linecolor='gray', ax=axes[1],
            xticklabels=labels, yticklabels=labels, cbar_kws={'fraction': 0.046, 'pad': 0.04})
axes[1].set_title('MI (Symbol-wise, a=2)')

plt.tight_layout()
plt.savefig(f"{out_dir}/Distances_Coded_n7.png")
plt.close()

# -------------------------------------------------------------------------------------------------
# 2. Mapa de Calor Bruto (Raw Coordinates) 16x7 com Seaborn
# -------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Mapeamento Direto das Dimensões: BMI vs MI (16 x 7)', fontsize=16, fontweight='bold')

dim_labels = [f"Dim {d+1}" for d in range(n)]

sns.heatmap(symbols_bmi, annot=True, fmt=".2f", cmap="vlag", linewidths=0.5, linecolor='gray', ax=axes[0],
            xticklabels=dim_labels, yticklabels=labels, cbar_kws={'fraction': 0.046, 'pad': 0.04})
axes[0].set_title('BMI (Bit-wise)')

sns.heatmap(symbols_mi, annot=True, fmt=".2f", cmap="vlag", linewidths=0.5, linecolor='gray', ax=axes[1],
            xticklabels=dim_labels, yticklabels=labels, cbar_kws={'fraction': 0.046, 'pad': 0.04})
axes[1].set_title('MI (Symbol-wise)')

plt.tight_layout()
plt.savefig(f"{out_dir}/Raw_Heatmap_Coded_n7.png")
plt.close()

# -------------------------------------------------------------------------------------------------
# 3. Projeção t-SNE (2D)
# -------------------------------------------------------------------------------------------------
from sklearn.manifold import TSNE

def normalize_tsne(coords):
    """
    Aplica a mesma normalização de energia realizada pela camada EnergyNormalization da rede:
    Centraliza os pontos e garante que a energia média por símbolo E[||x||^2] seja 1.
    """
    center = coords - np.mean(coords, axis=0, keepdims=True)
    energy_avg = np.mean(np.sum(np.square(center), axis=-1))
    return center / np.sqrt(energy_avg)

# Para t-SNE com n=16, a perplexidade deve ser menor que 16. Vamos usar 5.
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
pca_bmi = tsne.fit_transform(symbols_bmi)
pca_bmi = normalize_tsne(pca_bmi)

tsne = TSNE(n_components=2, perplexity=5, random_state=42)
pca_mi = tsne.fit_transform(symbols_mi)
pca_mi = normalize_tsne(pca_mi)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Projeção t-SNE 2D da Constelação 7D: BMI vs MI', fontsize=16, fontweight='bold')

for ax, pca_symbols, title in zip(axes, [pca_bmi, pca_mi], ['BMI (Bit-wise)', 'MI (Symbol-wise)']):
    ax.scatter(pca_symbols[:, 0], pca_symbols[:, 1], c='b', s=50)
    for i, label in enumerate(labels):
        ax.annotate(label, (pca_symbols[i, 0], pca_symbols[i, 1]), textcoords="offset points", xytext=(0, 5), ha='center')
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
    
    lim_max = np.max(np.abs(pca_symbols)) + 0.5
    ax.set_xlim(-lim_max, lim_max)
    ax.set_ylim(-lim_max, lim_max)

plt.tight_layout()
plt.savefig(f"{out_dir}/tSNE_Coded_n7.png")
plt.close()

# -------------------------------------------------------------------------------------------------
# 4. Violin Plot (Distância Hamming vs Euclidiana)
# -------------------------------------------------------------------------------------------------
data = []
for i in range(M):
    for j in range(i + 1, M):
        h_dist = sum(c1 != c2 for c1, c2 in zip(labels[i], labels[j]))
        data.append({'Modelo': 'BMI', 'Hamming': h_dist, 'Euclidiana': dist_bmi[i, j]})
        data.append({'Modelo': 'MI', 'Hamming': h_dist, 'Euclidiana': dist_mi[i, j]})

df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data=df, x='Hamming', y='Euclidiana', hue='Modelo', split=True, inner="quart", ax=ax, palette="muted")
ax.set_title('Correlação: Distância de Hamming vs Distância Euclidiana (Autoencoder 7,4)', fontsize=14, fontweight='bold')
ax.set_xlabel('Distância de Hamming')
ax.set_ylabel('Distância Euclidiana no Espaço 7D')
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(f"{out_dir}/Violin_Coded_n7.png")
plt.close()

print(f"Visualizações aprimoradas Coded n=7 geradas em {out_dir}")
