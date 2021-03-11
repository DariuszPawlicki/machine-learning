import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, data_dimension, hidden_size, dropout, batch_norm):
        super(Discriminator, self).__init__()
        
        self.network = nn.Sequential()

        for layer, layer_output in enumerate(hidden_size):
            if layer == 0:
                layer_input = data_dimension
            else:
                layer_input = hidden_size[layer-1]           
            
            self.network.add_module("affine_{}".format(layer+1), nn.Linear(layer_input, layer_output))
            
            if batch_norm == True:
                self.network.add_module("batch_norm_{}".format(layer+1), nn.BatchNorm1d(layer_output))
                
            self.network.add_module("dropout_{}".format(layer+1), nn.Dropout(p=dropout))
            self.network.add_module("relu_{}".format(layer+1), nn.LeakyReLU(0.2))
        
        self.network.add_module("affine_{}".format(len(hidden_size)+1), nn.Linear(hidden_size[-1], 1))
        self.network.add_module("output", nn.Sigmoid())
    

    def forward(self, x):
        return self.network(x)        


class Generator(nn.Module):
    def __init__(self, data_dimension, latent_dimension, hidden_size, batch_norm):
        super(Generator, self).__init__()
        
        self.network = nn.Sequential()        

        for layer, layer_output in enumerate(hidden_size):
            if layer == 0:
                layer_input = latent_dimension
            else:
                layer_input = hidden_size[layer-1]
            
            self.network.add_module("affine_{}".format(layer+1), nn.Linear(layer_input, layer_output))

            if batch_norm == True:
                self.network.add_module("batch_norm_{}".format(layer+1), nn.BatchNorm1d(layer_output))

            self.network.add_module("relu_{}".format(layer+1), nn.LeakyReLU(0.2))
        
        self.network.add_module("affine_{}".format(len(hidden_size)+1), nn.Linear(hidden_size[-1], data_dimension))
        self.network.add_module("output", nn.Tanh())


    def forward(self, x):
        return self.network(x)


class GAN(nn.Module):
    def __init__(self, data_dimension, latent_dimension, discriminator_size, generator_size, dropout=0.5, batch_norm=False):
        """   
        Input size is size of image flattened to vector, 
        for example 20x20 image gives 400x1 vector.

        Discriminator and generator size are tuples with numbers
        which are sizes of subsequent layers, for example
        (100, 100, 100) gives 3 layers with 100 neurons in each layer.

        k - number of iterations when only discriminator is learning
        """
        super(GAN, self).__init__()

        self.data_dimension = data_dimension

        if latent_dimension is None:
            self.latent_dimension = data_dimension
        else:
            self.latent_dimension = latent_dimension
        
        self.discriminator = Discriminator(self.data_dimension, discriminator_size, dropout=dropout, batch_norm=batch_norm)
        self.generator = Generator(self.data_dimension, self.latent_dimension, generator_size, batch_norm=batch_norm)
    

    def forward(self, x, generator=False):
        if generator == False:
            return self.discriminator(x)
        else:
            return self.generator(x)


    def discriminate(self, x):
        discr_out = self.discriminator(x)
        
        return np.where(discr_out >= 0.5, 1, 0)
    

    def generate(self, size):
        noise = torch.randn(size, self.latent_dimension)

        return self.generator(noise)