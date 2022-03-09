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
#       Descripcion :  Script que permite entrenar un modelo de ASR QuartsNet15x5
#                      Basado en el proyecto del Dr. Carlos Mena  https://github.com/cadia-lvl/samromur-asr/blob/n5_samromur/

# **************************************

#Imports

import sys
import re
import os

#Importing NeMo Modules
import nemo
import nemo.collections.asr as nemo_asr

########################################################################
#Input Parameters

NUM_GPUS=1

NUM_EPOCHS = 80

data_dir= "./data/"

EXPERIMENT_PATH = data_dir + 'experiments/'
#Model Architecture
config_path = data_dir + 'quartznet_15x5.yaml'

#Path to our training manifest
train_manifest = data_dir + 'manifests/train.json'

#Path to our validation manifest.
#The development portion in this case.
dev_manifest = data_dir + 'manifests/dev.json'

########################################################################
# Reading Model definition
from ruamel.yaml import YAML

yaml = YAML(typ='safe')
with open(config_path) as f:
    model_definition = yaml.load(f)
#ENDWITH

########################################################################
# Creating the trainer

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin

#Create a lightning log object
tb_logger = pl_loggers.TensorBoardLogger(save_dir=EXPERIMENT_PATH,name="lightning_logs")

#Create the trainer object
#trainer = pl.Trainer(gpus=NUM_GPUS, max_epochs=NUM_EPOCHS,logger=tb_logger,strategy="ddp")
trainer = pl.Trainer(gpus=NUM_GPUS, max_epochs=NUM_EPOCHS,logger=tb_logger,strategy=DDPPlugin(find_unused_parameters=False))

########################################################################
#Adjusting model parameters
from omegaconf import DictConfig

#Passing the path of the train manifest to the model
model_definition['model']['train_ds']['manifest_filepath'] = train_manifest
#Specifying the number of jobs of the training process

#Passing the path of the test manifest to the model
model_definition['model']['validation_ds']['manifest_filepath'] = dev_manifest

#Specifying the Learning Rate
model_definition['model']['optim']['lr'] = 0.005

#Specifying the Weight Decay
model_definition['model']['optim']['weight_decay'] = 0.0001

#Specifying the Dropout
model_definition['dropout']=0.2

#Specifying number of repetitions
model_definition['repeat']=1


########################################################################
#Creating the ASR system which is a NeMo object
nemo_asr_model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(model_definition['model']), trainer=trainer)

# Update vocabulary to spanish
nemo_asr_model.change_vocabulary(
    new_vocabulary=
    [
        ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
        'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'", "!"
    ] +
    [  # Append additional spanish characters
        'á', 'é', 'í', 'ñ', 'ó', 'ú', 'ü', '¿', '¡'
    ]
)
########################################################################
#START TRAINING!
trainer.fit(nemo_asr_model)

########################################################################
#Saving the Model

#Calculating the current date and time to label the checkpoint
from datetime import datetime
time_now=str(datetime.now())
time_now=time_now.replace(" ","_")

#Creating the Checkpoint directory
dir_checkpoints=os.path.join(EXPERIMENT_PATH,"CHECKPOINTS")
name_checkpoints= "model_weights_"+time_now+".ckpt"
name_checkpoints_nemo= "model_weights_"+time_now+".nemo"
if not os.path.exists(dir_checkpoints):
	os.mkdir(dir_checkpoints)
#ENDIF

#Save the checkpoint
path_checkpoint=os.path.join(dir_checkpoints, name_checkpoints)
path_checkpoint_nemo=os.path.join(dir_checkpoints, name_checkpoints_nemo)
nemo_asr_model.save_to(path_checkpoint_nemo)
nemo_asr_model.save_to(path_checkpoint)

########################################################################
#Write the path to the last checkpoint in an output file
file_path=os.path.join(EXPERIMENT_PATH,"final_model.path")
file_checkpoint=open(file_path,'w')
file_checkpoint.write(path_checkpoint)
file_checkpoint.close()

print("\nINFO: Final Checkpoint in "+path_checkpoint)

########################################################################

print("\nINFO: MODEL SUCCESFULLY TRAINED!")

########################################################################

