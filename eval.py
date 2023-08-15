from fastai.vision.all import *
import torch.backends.cudnn as cudnn
import utils
import argparse
from box import Box


parser = argparse.ArgumentParser(
    prog="Model Evaluation",
    description="Evaluates models trained with train.py"
)

parser.add_argument(
    '--data_path',
    '-d',
    type=str,
    help='Directory that contains model weights and config.',
    default='casia'
)
parser.add_argument(
    '--model_dir',
    '-i',
    type=str,
    help='Directory that contains model weights and config.',
    required=True
)
parser.add_argument(
    '--output_dir',
    '-o',
    help="Output directory where generated graphs and metrics are written",
    type=str,
    default='eval'
)

args = parser.parse_args()

# Open train config
with open(os.path.join(args.model_dir, 'train_config.yaml'), 'r') as file:
    train_config = Box.from_yaml(file)
    
with open('eval_config.yaml', 'r') as file:
    eval_config = Box.from_yaml(file)

# The following settings guarantee reproducibility
np.random.seed(config.random_seed) # cpu vars
torch.manual_seed(config.random_seed) # cpu  vars
random.seed(config.random_seed) # Python
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed) # gpu vars
    cudnn.deterministic = True
    cudnn.benchmark = False 