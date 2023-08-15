from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback
from fastai.callback.schedule import SaveModelCallback
import utils
import datetime
import torch
import random
import wandb
import torch.backends.cudnn as cudnn
import timm
import timm.optim

config = utils.openConfig()

# The following settings guarantee reproducibility
np.random.seed(config.random_seed) # cpu vars
torch.manual_seed(config.random_seed) # cpu  vars
random.seed(config.random_seed) # Python
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed) # gpu vars
    cudnn.deterministic = True
    cudnn.benchmark = False   
    

training_id =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
final_path = os.path.join(config.output_path, training_id)

# print('-----------------------------')
# print(f'Beginning training with id {training_id}!')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataframe = utils.getDataframe(**config)

dataloaders = ImageDataLoaders.from_df(dataframe, valid_pct=0, valid_col='is_test', seed=config.random_seed, 
                         bs=config.batch_size, shuffle_train=config.shuffle_train,
                         device=device, item_tfms=Resize(config.resize))

train_count = np.unique(dataloaders.train_ds.items['label'], return_counts=True)[1].mean()
test_count = np.unique(dataloaders.valid_ds.items['label'], return_counts=True)[1].mean()
print(f"Split {train_count} images for train and {test_count} for test")

# model = timm.create_model(**config.model)

# opt_func = Adam
# loss_func = CrossEntropyLossFlat()
callbacks = [WandbCallback(**config.wandb), 
               SaveModelCallback(**config.save_model), 
               ReduceLROnPlateau(**config.reduce_lr)
               ]

learn = vision_learner(dls=dataloaders, 
                       arch=config.model_name, 
                       metrics=accuracy,
                    #    loss_func=loss_func,
                    #    opt_func=opt_func,
                       cbs=callbacks,
                       path=final_path,
                       model_dir='',
                    #    lr=config.learning_rate
                       )

os.makedirs(final_path, exist_ok=True)

# Write config to output folder
with open(os.path.join(final_path, 'config.yaml'), 'w') as file:
    yaml.dump(config, file)

wandb.init(project="ml-iris-vit", config=config, dir=final_path, 
           sync_tensorboard=config.sync_tensorboard)


learn.fine_tune(config.epochs, freeze_epochs=config.freeze_epochs, start_epoch=config.start_epoch)

wandb.finish()