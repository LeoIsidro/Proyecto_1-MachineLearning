import numpy as np
import pandas as pd
from  skimage.io import imread, imshow
from pathlib import Path
import pywt
import pywt.data



class Dataset(object):
    def __init__(self,ruta):
        self.ruta = ruta

    def load(self):
        df = pd.read(self.ruta)
        return df


def Get_Feacture(picture, cortes):
  LL = picture
  for i in range(cortes):
     LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
  return LL.flatten().tolist()


path='./imagenes_1/'
vectores_caracteristicos=[]
entries = Path(path)
for entry in entries.iterdir():
  imagen = path + entry.name
  picture = imread(imagen)
  vectores_caracteristicos.append(Get_Feacture(picture,2))



  

