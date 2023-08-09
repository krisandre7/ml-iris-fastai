import timm 
from fastai.vision.all import *
import utils
import datetime


config = utils.openConfig()

training_id =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
final_path = os.path.join(config.output_path, training_id)
os.makedirs(final_path, exist_ok=True)
print('-----------------------------')
print(f'Beginning training with id {training_id}!')

    
utils.getDataset(**config)
