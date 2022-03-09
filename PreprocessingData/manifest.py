# ----------------------------------------------------------------------------------------------------------------------------------------------
#                                                  ___   _______    ___       ___ 
#                                                / _ | / __/ _ \  / _ \__ __/ _ \
#                                               / __ |_\ \/ , _/ / ___/ // / ___/
#                                              /_/ |_/___/_/|_| /_/   \_, /_/    
#                                                                    /___/                                                    
# 
#                                            INST. DE INV. EN MATEMATICAS APLICADAS Y SISTEMAS
#                                        Desarrollo de sistemas inteligentes usando deep learning
# ----------------------------------------------------------------------------------------------------------------------------------------------

#       Autores:  - Adolfo Patricio Barrero Olguin
#                 - Ingrid Pamela Ruiz Puga

#       Actualizacion : Marzo 2022

# **************************************
#       Descripcion :  Script que permite crear los archivos manifest.json del conjunto de datos de Mozilla Common Voice
        #Ejecutar python3 manifest.py 

# **************************************

# --- Librerias --- #
import json
import librosa
import sys
import pandas as pd
import os

# --- Directorios --- #
#data_dir = sys.argv[1]
data_dir = 'data/cv-corpus-7.0-2021-07-21/es/'
#wav_path = sys.argv[2]
wav_path = 'clips_wav/'

# Function to build a manifest
def build_manifest(transcripts_path, file_name, wav_path):
    metadata_file = pd.read_csv(transcripts_path, usecols= ['client_id','path',	'sentence',	
                                                            'up_votes',	'down_votes','age','gender',
                                                            'accent','locale','segment'], sep='\t', 
                                                            encoding="utf8")

    with open(os.path.join(data_dir, file_name), 'w') as fout:
        for line in range(metadata_file.shape[0]):
            # Lines look like this:
            # <s> transcript </s> (fileID)
            transcript = metadata_file.sentence[line].lower()
            transcript = transcript.replace('<s>', '').replace('</s>', '')
            transcript = transcript.strip()

            #file_id = line[line.find('(')+1 : -2]  # e.g. "cen4-fash-b"
            audio_path = os.path.join(
                data_dir, wav_path,
                metadata_file.path[line][:-3]+'wav')
            try:
                duration = librosa.core.get_duration(filename=audio_path)
            except:
                continue    
                # Write the metadata to the manifest
            metadata = {
                "audio_filepath": audio_path,
                "duration": duration,
                "text": transcript
            }
            json.dump(metadata, fout)
            fout.write('\n')
        fout.close()  


if __name__ == '__main__': 

# Building Manifests
    print("******  Start Manifest   ******")
    train_transcripts = data_dir + 'train.tsv'
    train_manifest = data_dir + "'manifest/train_manifest.json"
    if not os.path.isfile(train_manifest):
        build_manifest(train_transcripts, "train_manifest.json", wav_path)
        print("******  Training manifest created.   ******")

    test_transcripts = data_dir + 'test.tsv'
    test_manifest = data_dir + "manifest/test_manifest.json"
    if not os.path.isfile(test_manifest):
        build_manifest(test_transcripts, "test_manifest.json", wav_path)
        print("******  Test manifest created.   ******")

    dev_transcripts = data_dir + 'dev.tsv'
    dev_manifest = data_dir + "manifest/dev_manifest.json"
    if not os.path.isfile(dev_manifest):
        build_manifest(dev_transcripts, "dev_manifest.json", wav_path)
        print("******  Dev manifest created.   ******")

    print("***Done***")