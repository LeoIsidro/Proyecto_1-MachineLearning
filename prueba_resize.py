from PIL import Image
import os

# Lista de rutas de archivo de las imágenes que deseas cargar
lista_de_imagenes = ["./0010001.png", "./0010002.png", "./0010006.png","./0010010.png"]

# Tamaño deseado para todas las imágenes
ancho_deseado = 300
alto_deseado = 200

# Carpeta de destino para las imágenes redimensionadas
carpeta_destino = "imagenes_1"

# Crea la carpeta de destino si no existe
if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)

# Recorre la lista de imágenes y redimensiona y guarda cada una
for ruta_imagen in lista_de_imagenes:
    imagen = Image.open(ruta_imagen)
    
    # Redimensiona la imagen al tamaño deseado con interpolación ANTIALIAS
    imagen_redimensionada = imagen.resize((ancho_deseado, alto_deseado), Image.ADAPTIVE)
    
    # Obtiene el nombre de archivo sin la ruta
    nombre_archivo = os.path.basename(ruta_imagen)
    
    # Define la ruta de archivo de destino en la carpeta de destino
    ruta_destino = os.path.join(carpeta_destino, nombre_archivo)
    
    # Guarda la imagen redimensionada en la carpeta de destino
    imagen_redimensionada.save(ruta_destino)

