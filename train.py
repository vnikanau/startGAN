from typing import Tuple
import math

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DistributedSampler

from loader import Dataset
from models import Discriminator, Generator
from utils import generate_even_data, convert_float_matrix_to_int_list

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

def train(
    max_int: int = 128,
    batch_size: int = 16,
    training_steps: int = 500,
    learning_rate: float = 0.001,
    print_output_every_n_steps: int = 10,
) -> Tuple[nn.Module]:
    """Trains the even GAN

    Args:
        max_int: The maximum integer our dataset goes to.  It is used to set the size of the binary
            lists
        batch_size: The number of examples in a training batch
        training_steps: The number of steps to train on.
        learning_rate: The learning rate for the generator and discriminator
        print_output_every_n_s The number of training steps before we print generated output

    Returns:
        generator: The trained generator model
        discriminator: The trained discriminator model
    """
    input_length = int(math.log(max_int, 2))

    folder = "/home/v_nikonov/projects/datasets/pets/"
    dataset = Dataset(folder, 256, transparent=False)

    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    loader = cycle(dataloader)

    # Models
    generator = Generator(input_length)
    discriminator = Discriminator(input_length)

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=learning_rate
    )

    # / home / v_nikonov / projects / datasets

    # loss
    loss = nn.BCELoss()

    for i in range(training_steps):
        # zero the gradients on each iteration
        generator_optimizer.zero_grad()

        # Create noisy input for generator
        # Need float type instead of int
        noise = torch.randint(0, 2, size=(batch_size, input_length)).float()
        generated_data = generator(noise)


        # Train the generator
        # We invert the labels here and don't train the discriminator because we want the generator
        # to make things the discriminator classifies as true.
        generator_discriminator_out = discriminator(generated_data)
        true_labels = [1] * batch_size
        true_labels = torch.tensor(true_labels).float()

        true_labels_ = true_labels.unsqueeze(1)

        generator_loss = loss(generator_discriminator_out, true_labels_)
        generator_loss.backward()
        generator_optimizer.step()

        # Train the discriminator on the true/generated data
        discriminator_optimizer.zero_grad()

        # Generate examples of even real data
        # true_labels, true_data = generate_even_data(max_int, batch_size=batch_size)
        true_data = next(loader)

        true_data = torch.tensor(true_data).float()
        true_discriminator_out = discriminator(true_data)
        true_discriminator_loss = loss(true_discriminator_out, true_labels_)

        # add .detach() here think about this
        generator_discriminator_out = discriminator(generated_data.detach())

        torchz = torch.zeros(batch_size)
        torchz = torchz.unsqueeze(1)

        generator_discriminator_loss = loss(
            generator_discriminator_out, torchz
        )
        discriminator_loss = (
            true_discriminator_loss + generator_discriminator_loss
        ) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()
        if i % print_output_every_n_steps == 0:
            print(convert_float_matrix_to_int_list(generated_data))

    torch.save(generator, "2021.pt")
    from torch.utils.mobile_optimizer import optimize_for_mobile
    scripted_module = torch.jit.script(generator)
    # Export mobile interpreter version model (compatible with mobile interpreter)
    optimized_scripted_module = optimize_for_mobile(scripted_module)
    optimized_scripted_module._save_for_lite_interpreter("2021.ptl")

    return generator, discriminator


if __name__ == "__main__":
    train()
