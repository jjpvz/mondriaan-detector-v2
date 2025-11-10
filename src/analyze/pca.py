from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

def plot_descriptor_pca(descriptor_list, class_labels, descriptor_name="Descriptor", n_components=2):
    X = np.array(descriptor_list)
    
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    classes, class_indices = np.unique(class_labels, return_inverse=True)
    
    plt.figure(figsize=(6,5))
    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=class_indices, cmap='Set1', alpha=0.7)
    
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA of {descriptor_name}")
    
    handles = []
    for i, class_name in enumerate(classes):
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=plt.cm.Set1(i / len(classes)),
                                  markersize=8, label=class_name))
    plt.legend(handles=handles)
    
    plt.tight_layout()
    plt.show()
