import os
from torch.backends import cudnn
import glob
import numpy as np
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.utils import save_image
from Generator import Generator

cudnn.benchmark = True
model_save_dir = "./checkpoint"
result_dir = "./results"


class Test(object):
    def __init__(self, data_loader):
        self.loader = data_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.build_model()

    def restore_model(self):
        """Restore the trained generator"""
        print('Loading the trained models')
        G_path = os.path.join(model_save_dir, 'net_G_Face.pth')
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator()
        self.G.to(self.device)

    def test(self):
        # Load the trained generator.
        self.restore_model()
        data_loader = self.loader
        self.G.eval()
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                sketch, real = data['sketch'], data['real']
                sketch = sketch.to(self.device)
                real = real.to(self.device)
                result_path = os.path.join(result_dir, '{}images.jpg'.format(i + 1))
                # save_image(self.denorm(self.G(sketch)[0]), result_path, nrow=1, padding=0)
                # save_image(self.denorm(real), result_path, nrow=1, padding=0)
                save_image(torch.cat([self.denorm(sketch), self.denorm(self.G(sketch)[0]),
                                      self.denorm(real)],
                                     dim=3), result_path, nrow=1, padding=0)

                print('Saved real and fake images into {}...'.format(result_path))


def make_dataloaders(real_path, sketch_path, batch_size=1, n_workers=4,
                     pin_memory=True):  # A handy function to make our dataloaders
    dataset = MyDataset(real_path, sketch_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader


if __name__ == '__main__':
    path_real = "./test/real"
    path_sketch = "./test/sketch"

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(path_real):
        os.makedirs(path_real)
    if not os.path.exists(path_sketch):
        os.makedirs(path_sketch)
    paths_real = np.array(glob.glob(path_real + "/*.jpg"))  # Grabbing all the image file names
    paths_sketch = np.array(glob.glob(path_sketch + "/*.jpg"))

    test_dl = make_dataloaders(real_path=paths_real, sketch_path=paths_sketch, batch_size=5)
    myTest = Test(test_dl)
    myTest.test()
