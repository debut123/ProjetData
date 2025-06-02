import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data1 = pd.read_csv("C:/Users/guill/Downloads/data1.csv",sep=",",index_col=0);
data1.head()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data1)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
var1, var2 = pca.explained_variance_ratio_
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], edgecolor='k')
for i, city in enumerate(data1.index):
    plt.text(X_pca[i, 0] + 0.03, X_pca[i, 1] + 0.03, city)

plt.xlabel(f'Composante Principale 1 ({var1*100:.1f} % de variance)')
plt.ylabel(f'Composante Principale 2 ({var2*100:.1f} % de variance)')
plt.title('Projection des villes sur les deux premières composantes principales')
plt.grid(True)
plt.show()


nlignes, ncol = data1.shape
vpropres = pca.explained_variance_
sqrt_vpropres = np.sqrt(vpropres)
corvar = np.zeros((ncol, pca.n_components_))

for k in range(pca.n_components_):
    corvar[:, k] = pca.components_[k, :] * sqrt_vpropres[k]

fig, ax = plt.subplots(figsize=(6,6.5))
an = np.linspace(0, 2 * np.pi, 100)
ax.plot(np.cos(an), np.sin(an), '--', linewidth=0.5)

for i in range(corvar.shape[0]):
    ax.arrow(0, 0,corvar[i, 0],corvar[i, 1],head_width=0.03,length_includes_head=True,color='b')
    ax.text(corvar[i, 0] * 1.15,corvar[i, 1] * 1.15,data1.columns[i],ha='center',va='center')


ax.set_xlabel(f"PC1 ({var1*100:.1f}% de variance)")
ax.set_ylabel(f"PC2 ({var2*100:.1f}% de variance)")
ax.set_title('Cercle de corrélation')
plt.grid(True)
ax.axis('equal')
plt.show()


vpropres = pca.explained_variance_
sqrt_vpropres = np.sqrt(vpropres)
corvar = np.zeros((data1.shape[1], pca.n_components_))
for k in range(pca.n_components_):
    corvar[:, k] = pca.components_[k, :] * sqrt_vpropres[k]


factor = np.max(np.abs(X_pca))

plt.figure(figsize=(8, 8))

plt.scatter(X_pca[:, 0], X_pca[:, 1], edgecolor='k')
for i, city in enumerate(data1.index):
    plt.text(X_pca[i, 0] + 0.03, X_pca[i, 1] + 0.03, city)


cer = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(cer) * factor, np.sin(cer) * factor, '--', color='grey')

for i, var in enumerate(data1.columns):
    x, y = corvar[i, 0] * factor, corvar[i, 1] * factor
    plt.arrow(0, 0, x, y, head_width=0.03*factor, length_includes_head=True, color='b')
    plt.text(x * 1.1, y * 1.1, var, color='r')

plt.xlabel(f'PC1 ({var1*100:.1f}% de variance)')
plt.ylabel(f'PC2 ({var2*100:.1f}% de variance)')
plt.title('Projection des villes et cercle de corrélation')
plt.grid(True)
plt.axis('equal')
plt.show()