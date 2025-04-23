import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carpetas con las imágenes a procesar
CARPETA_WP = 'white_patch'
CARPETA_CROMATICAS = 'coord_cromaticas'


def cargar_imagenes(carpeta):
    
    imagenes = []
    nombres = []
    
    for archivo in os.listdir(carpeta):
        if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            ruta = os.path.join(carpeta, archivo)
            img = cv2.imread(ruta)
            # Convertir de BGR a RGB para visualización correcta
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imagenes.append(img)
            nombres.append(archivo)
            
    return imagenes, nombres


def coord_cromaticas(imagen):
    
    # Necesitamos trabajar en flotante para la división
    img_float = imagen.astype(np.float32)
    
    # Sumamos los tres canales RGB
    suma = np.sum(img_float, axis=2, keepdims=True)
    
    # Evitar división por cero en píxeles negros
    suma[suma == 0] = 1
    
    # Normalizar cada canal por la suma total
    return img_float / suma


def white_patch(imagen):
  
    img_float = imagen.astype(np.float32)
    
    # Buscar el valor máximo en cada canal
    max_rgb = np.max(img_float, axis=(0, 1))
    
    # Prevenir división por cero
    max_rgb[max_rgb == 0] = 1
    
    # Normalizar y escalar a 8 bits
    img_corregida = img_float / max_rgb * 255.0
    
    # Asegurar valores válidos y convertir a entero
    return np.clip(img_corregida, 0, 255).astype(np.uint8)


def mostrar_resultados(carpeta):
    
    print(f"\nProcesando imágenes en '{carpeta}'...")
    
    # Cargar todas las imágenes de la carpeta
    imagenes, nombres = cargar_imagenes(carpeta)
    
    # Procesar cada imagen
    for img, nombre in zip(imagenes, nombres):
        # Aplicar los dos algoritmos
        img_cromatica = coord_cromaticas(img)
        img_wp = white_patch(img)
        
        # Mostrar los resultados en una figura con tres paneles
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Imagen original
        axs[0].imshow(img)
        axs[0].set_title(f'Original: {nombre}')
        
        # Coordenadas cromáticas
        axs[1].imshow(img_cromatica)
        axs[1].set_title('Coordenadas cromáticas')
        
        # White Patch
        axs[2].imshow(img_wp)
        axs[2].set_title('White Patch')
        
        # Quitar ejes
        for ax in axs:
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()
        
        # Análisis básico
        print(f"Análisis de '{nombre}':")
        
        # Revisar si hay un blanco puro
        if np.max(img, axis=(0, 1)).min() < 250:
            print("  Nota: No se detectó un blanco puro en la imagen.")
        
        # Verificar saturación
        if np.any(img_wp == 255):
            print("  Nota: Hay píxeles saturados después de la corrección.")
            
        print()  # Espacio entre imágenes


def main():
    
    print("Procesamiento de imágenes - Técnicas de corrección de color")
    print("-" * 55)
    
    # Procesar ambas carpetas
    mostrar_resultados(CARPETA_WP)
    mostrar_resultados(CARPETA_CROMATICAS)
    
    print("Procesamiento completado.")


# Punto de entrada estándar
if __name__ == '__main__':
    main()