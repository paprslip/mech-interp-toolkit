import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from jaxtyping import Float
from typing import Tuple

def pca(
    X: Float[np.ndarray, "batch d_model"]
):

    colors = []
    for i in range(len(X)//2):
        colors.append('red')
    for j in range(i + 1, len(X)):
        colors.append('blue')

    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(Z[:, 0], Z[:, 1], 
                c=colors, alpha=0.6, s=100)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('Feature Space Visualization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    #plt.show()
    
    return fig, pca
