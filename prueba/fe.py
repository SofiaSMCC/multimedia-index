import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import heapq
import matplotlib.pyplot as plt

# Cargar modelo ResNet-50 preentrenado
resnet50 = models.resnet50(pretrained=True)
resnet50.eval()  # Modo de evaluación

# Eliminar la capa de clasificación
feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1])

# Definir transformaciones de imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionar la imagen
    transforms.ToTensor(),  # Convertir a tensor
    transforms.Normalize(  # Normalización basada en ImageNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Función para extraer el vector de características
def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # Agregar una dimensión para el batch

    # Extraer características
    with torch.no_grad():
        features = feature_extractor(input_tensor)

    # Aplanar las características para obtener el vector final
    feature_vector = features.squeeze().numpy()
    return feature_vector


# Búsqueda KNN con Cola de Prioridad para imágenes
def knn_priority_queue_images(data_features, query_feature, k):
    heap = []

    # Calcular la distancia y mantener solo los k vecinos más cercanos
    for idx, feature in enumerate(data_features):
        dist = np.linalg.norm(query_feature - feature)  # Distancia Euclidiana

        if len(heap) < k:
            heapq.heappush(heap, (-dist, idx))  # Guardar distancias negativas
        else:
            heapq.heappushpop(heap, (-dist, idx))  # Mantener solo K vecinos más cercanos

    # Extraer los índices de los k vecinos más cercanos
    nearest_neighbors = [idx for _, idx in heap]
    return nearest_neighbors


# Ruta de la imagen de consulta
query_image_path = 'pokemon/Abra/00000000.png'  # Especificar la ruta completa de la imagen de consulta
query_feature = extract_features(query_image_path)

# Cargar todas las imágenes de la carpeta
image_folder = 'pokemon/Abra'
image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith(('.jpg', '.jpeg', '.png'))]

# Extraer las características de todas las imágenes en la carpeta
data_features = [extract_features(image_path) for image_path in image_paths]

# Número de vecinos más cercanos
k = 5  # Número de vecinos más cercanos

# Buscar las k imágenes más similares usando KNN
similar_images = knn_priority_queue_images(data_features, query_feature, k)

# Mostrar los resultados
print(f"Imagen consulta: {query_image_path}")
print("5 imágenes más similares:")
for idx in similar_images:
    print(f"- {image_paths[idx]}")
print("-" * 50)


# Búsqueda por Rango para imágenes
def range_search_images(data_features, query_feature, radius):
    neighbors = []

    # Calcular las distancias y devolver los objetos dentro del radio
    for idx, feature in enumerate(data_features):
        dist = np.linalg.norm(query_feature - feature)  # Distancia Euclidiana
        if dist <= radius:
            neighbors.append((dist, idx))

    # Ordenar los resultados por distancia
    neighbors.sort(key=lambda x: x[0])
    return neighbors


# Ejemplo de uso para búsqueda por rango
radius = 0.5  # Radio de búsqueda
neighbors = range_search_images(data_features, query_feature, radius)
print("Imágenes dentro del radio de búsqueda:")
for _, idx in neighbors:
    print(f"- {image_paths[idx]}")
print("-" * 50)


# Función para graficar la distribución de distancias entre imágenes
def plot_distance_distribution_images(data_features, query_feature):
    distances = []
    for feature in data_features:
        dist = np.linalg.norm(query_feature - feature)
        distances.append(dist)

    # Graficar el histograma de distancias
    plt.hist(distances, bins=30, edgecolor='black')
    plt.title('Distribución de Distancias (Imágenes)')
    plt.xlabel('Distancia')
    plt.ylabel('Frecuencia')
    plt.show()


# Ejemplo de uso para graficar la distribución de distancias
plot_distance_distribution_images(data_features, query_feature)
