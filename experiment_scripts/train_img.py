from .. import dataio, loss_functions

from torch.utils.data import DataLoader
import configparser
import functools

p = configparser.ArgumentParser()

# General training parameters
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--num_epochs', type=int, default=10000)

p.add_argument('--epochs_per_save', type=int, default=25)
p.add_argument('--epochs_per_eval', type=int, default=1000)

opt = p.parse_args()

img_dataset = dataio.ImageFile('INSERT IMAGE FILE HERE')
coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=512, compute_diff='all')
img_resolution = (512, 512)


dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)
