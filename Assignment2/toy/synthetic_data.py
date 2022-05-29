import numpy as np
import torch
import random
import matplotlib.pyplot as plt



def create_toy_data():
    samples = torch.rand(10000, 50, 1)
    for sample in samples:
        i = np.random.randint(low=20, high=30)
        for j in range(i-5, i+6):
            sample[j] *= 0.1
    return samples


def toy_data():
    samples = create_toy_data()
    train = samples[0:6000, :, :]
    validation = samples[6000:8000, :, :]
    test = samples[8000:10000, :, :]

    return train, validation, test

def plot_toy_sample(samples):
    inds = random.sample(range(10000), 3)
    for i in inds:
        plt.plot(samples[i])
        plt.xlabel("Time")
        plt.ylabel("Signal Value")
        plt.title(f'Example {i} from the Synthetic data')
        plt.show()


# samples = create_toy_data()
# plot_toy_sample(samples)





