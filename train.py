from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback
from fastai.callback.schedule import SaveModelCallback
import utils
import datetime
import torch
import random

config = utils.openConfig()
np.random.seed(config.random_seed) # cpu vars
torch.manual_seed(config.random_seed) # cpu  vars
random.seed(config.random_seed) # Python
if torch.cuda.is_available():
    print('USING CUDA')
    torch.cuda.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed) # gpu vars
    torch.backends.cudnn.deterministic = True  #needed
    torch.backends.cudnn.benchmark = False

git config --global user.email "you@example.com"
  git config --global user.name "Your Name"

training_id =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
final_path = os.path.join(config.output_path, training_id)

os.makedirs(final_path, exist_ok=True)

# Write config to output folder
with open(os.path.join(final_path, 'config.yaml'), 'w') as file:
    yaml.dump(config, file)
print('-----------------------------')
print(f'Beginning training with id {training_id}!')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataframe = utils.getDataframe(**config)

dataloaders = ImageDataLoaders.from_df(dataframe, valid_pct=0, valid_col='is_test', seed=config.random_seed, 
                         bs=config.batch_size, shuffle_train=config.shuffle_train,
                         device=device)

train_count = np.unique(dataloaders.valid_ds.items['label'], return_counts=True)[1].mean()
test_count = np.unique(dataloaders.train_ds.items['label'], return_counts=True)[1].mean()
print(f'Split {train_count} images for train and {test_count} for test')

wandb = WandbCallback(**config.wandb)
model_save = SaveModelCallback(**config.save_model)
reduce_lr = ReduceLROnPlateau(**config.reduce_lr)
learn = vision_learner(dataloaders, resnet34, metrics=accuracy)

learn.fine_tune(config.epochs)