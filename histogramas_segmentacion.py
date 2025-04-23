import cv2
import numpy as np
import matplotlib.pyplot as plt


def mostrar_imagenes_grises():

    img1 = cv2.imread('img1_tp.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('img2_tp.png', cv2.IMREAD_GRAYSCALE)
    
    
    plt.figure(figsize=(10, 4))
    
    # Primera imagen
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('img1_tp.png')
    plt.axis('off')
    
    # Segunda imagen
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.title('img2_tp.png')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return img1, img2


def analizar_histogramas(img1, img2, bins=32):
    """Genera y compara los histogramas de dos imágenes"""
    plt.figure(figsize=(10, 4))
    
    # Histograma de la primera imagen
    plt.subplot(1, 2, 1)
    plt.hist(img1.flatten(), bins=bins, color='blue', alpha=0.7)
    plt.title('Histograma img1_tp.png')
    plt.xlabel('Nivel de gris')
    plt.ylabel('Cantidad de píxeles')
    
    # Histograma de la segunda imagen
    plt.subplot(1, 2, 2)
    plt.hist(img2.flatten(), bins=bins, color='green', alpha=0.7)
    plt.title('Histograma img2_tp.png')
    plt.xlabel('Nivel de gris')
    plt.ylabel('Cantidad de píxeles')
    
    plt.tight_layout()
    plt.show()
    
    # Análisis de los resultados: 
    # Las diferencias en los histogramas revelan distintas distribuciones de intensidad entre las imágenes o regiones analizadas. 
    # Histogramas distintos son útiles como características para clasificación, ya que estas variaciones pueden servir para 
    # diferenciar entre categorías o patrones. Por otro lado, los histogramas similares pueden indicar baja capacidad discriminativa 
    # entre los elementos comparados, lo que sugiere limitaciones para distinguirlos mediante este método. Finalmente, es importante 
    # destacar que los histogramas capturan información global sobre la frecuencia de los valores de intensidad, pero no reflejan la 
    # estructura espacial ni las relaciones contextuales dentro de los datos analizados.



def segmentar_por_color():
    
    # Cargar imagen a color
    img = cv2.imread('segmentacion.png')
    
    # Convertir de BGR (OpenCV) a RGB (para matplotlib)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Mostrar histogramas por canal de color
    canales = ['Rojo', 'Verde', 'Azul']
    colores = ['red', 'green', 'blue']
    
    plt.figure(figsize=(12, 4))
    for i, (nombre, color) in enumerate(zip(canales, colores)):
        plt.subplot(1, 3, i+1)
        plt.hist(img_rgb[..., i].ravel(), bins=32, color=color, alpha=0.7)
        plt.title(f'Canal {nombre}')
        plt.xlabel('Intensidad')
    
    plt.tight_layout()
    plt.show()
    
    
    # Agua: tonos azules oscuros
    mascara_agua = (img_rgb[..., 2] > 80) & (img_rgb[..., 0] < 100) & (img_rgb[..., 1] < 100)
    
    # Cielo: azul más claro
    mascara_cielo = (img_rgb[..., 2] > 150) & (img_rgb[..., 0] < 180) & (img_rgb[..., 1] > 120)
    
    # Tierra: tonos marrones/rojizos
    mascara_tierra = (img_rgb[..., 0] > 80) & (img_rgb[..., 1] > 40) & (img_rgb[..., 2] < 100)
    
    # Diccionario con las máscaras
    regiones = {
        'Agua': mascara_agua, 
        'Cielo': mascara_cielo, 
        'Tierra': mascara_tierra
    }
    
    # Visualizar y guardar cada región segmentada
    for nombre, mascara in regiones.items():
        # Crear copia para no modificar la original
        img_segmentada = img_rgb.copy()
        
        # Aplicar máscara (poner en negro lo que no corresponde a la región)
        img_segmentada[~mascara] = 0
        
        # Mostrar resultado
        plt.figure()
        plt.imshow(img_segmentada)
        plt.title(f'Segmentación: {nombre}')
        plt.axis('off')
        plt.show()
        
        # Guardar imagen (convertir de RGB a BGR para OpenCV)
        cv2.imwrite(f'region_{nombre.lower()}.png', 
                   cv2.cvtColor(img_segmentada, cv2.COLOR_RGB2BGR))
    
def main():
    """Función principal del programa"""
    print("-- Análisis de imágenes mediante histogramas --")
    
    # Cargar y mostrar imágenes en escala de grises
    img1, img2 = mostrar_imagenes_grises()
    
    # Analizar histogramas de las imágenes en escala de grises
    analizar_histogramas(img1, img2, bins=32)
    
    # Realizar segmentación por color
    segmentar_por_color()
    
if __name__ == '__main__':
    main()