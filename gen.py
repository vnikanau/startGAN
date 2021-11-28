from typing import Tuple
import math

import torch
import torch.nn as nn

from models import Discriminator, Generator
from utils import generate_even_data, convert_float_matrix_to_int_list
from torchvision.utils import save_image

def gen():
    batch_size: int = 16
    max_int: int = 128
    input_length = int(math.log(max_int, 2))

    model = torch.load("2000.pt")
    model.eval()
    noise = torch.randint(0, 2, size=(batch_size, input_length)).float()
    generated_data = model(noise)

    generated_data.shape #torch.Size([64,3,28,28])
    #img1 = generated_data[0] #torch.Size([3,28,28]
    # img1 = img1.numpy() # TypeError: tensor or list of tensors expected, got <class 'numpy.ndarray'>
    #save_image(img1, 'img1.png')
    print( generated_data )
    print("done")

if __name__ == "__main__":
    gen()
