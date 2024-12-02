import matplotlib.pyplot as plt
from Experiments.RTree_exp import Run_KnnRtree
from Experiments.HighD_exp import Run_KnnLSH
from Experiments.Sequential_exp import Run_KnnSequential, Run_RangeSearch
import numpy as np


# === Visualización ===

def plot_distance_distribution(data_features, query_feature):
    """Grafica la distribución de distancias entre imágenes."""
    distances = [np.linalg.norm(query_feature - feature) for feature in data_features]
    plt.hist(distances, bins=30, edgecolor='black')
    plt.title('Distribución de Distancias')
    plt.xlabel('Distancia')
    plt.ylabel('Frecuencia')
    plt.show()

# === Principal ===

def main():

    query_image_path = 'utils/2000_datos/00000000_charmander (3).png'
    feature_file = 'VectorFeatures/image_features_2000.npz'
    path = 'utils/2000_datos'

    # Búsqueda KNN sin indexación
    Run_RangeSearch(query_image_path, feature_file, path, radius=10)

    # Búsqueda por rango sin indexación
    Run_KnnSequential(query_image_path, feature_file, path, k=5)

    # Búsqueda KNN con R-tree
    Run_KnnRtree(query_image_path, feature_file, path, k=5)

    # Búsqueda KNN con Faiss
    Run_KnnLSH(query_image_path, feature_file, path, k=5)

if __name__ == "__main__":
    main()