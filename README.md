# ASR with Nemo
                                  ___   _______    ___       ___ 
                                 / _ | / __/ _ \  / _ \__ __/ _ \
                                / __ |_\ \/ , _/ / ___/ // / ___/
                               /_/ |_/___/_/|_| /_/   \_, /_/    
                                                     /___/                                                    
        
                           INST. DE INV. EN MATEMATICAS APLICADAS Y SISTEMAS
                       Desarrollo de sistemas inteligentes usando deep learning


En este repositorio encontraras los pasos y archivos necesarios para entrenar un modelo de Reconocimiento de voz.

Las librelias necesarias serán:

- Nemo (ver instalación)
- Pytorch
- Pytorch lightning
- libsndfile1
- ffmpeg
- Librosa
- Pandas

## Instalación de Nemo

Es necesario tener instalado Python 3, y posteriormente la instalacion general de Nemo es: 
  
  	$ pip install Cython
  	$ pip install nemo_toolkit['all']
  
Tambien será necesario Pytorch y en especifico 
 
	$ pip install pytorch-lightning
  

## Datos
Se utilizó el corpus Common voice de mozzila  Fuente: [https://commonvoice.mozilla.org/en/datasets]
En el idioma español con un total de 18GB y (130 GB en versión wav) con las siguientes caracteristicas:
- Total de horas: 739 HRS de audio.
- Número de voces: 79,398
- Edades: ( < 19  ) 6%
- 
          (19 - 29) 24% 
          
          (30 - 39) 13%  
          
          (40 - 49) 10%
          
          (50 - 59) 4%
          
          (60 - 69) 4%
          
          (70 - 79) 1%
          
- Género: Masculino 46%

          Femenino  16%

 
Female

## Fuentes
