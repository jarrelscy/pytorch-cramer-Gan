import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

activation = nn.LeakyReLU

class MLP_G(nn.Module):
    def __init__(self, input_size, noise_dim, num_chan, hid_dim, ngpu=1):
        super(MLP_G, self).__init__()
        self.ngpu = ngpu
        self.register_buffer('device_id', torch.zeros(1))
        main = nn.Sequential(
            # Z goes into a linear of size: hid_dim
            nn.Linear(noise_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, num_chan * input_size * input_size),
            nn.Sigmoid()
        )
        self.main = main
        self.num_chan = num_chan
        self.input_size = input_size
        self.noise_dim = noise_dim

    def forward(self, inputs):
        inputs = inputs.view(inputs.size(0), inputs.size(1))
        if isinstance(inputs.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
        else:
            output = self.main(inputs)
        return output.view(output.size(0), self.num_chan, self.input_size, self.input_size)


class MLP_D(nn.Module):
    def __init__(self, input_size, num_chan, hid_dim,out_dim=1, ngpu=1):
        super(MLP_D, self).__init__()
        self.ngpu = ngpu
        self.register_buffer('device_id', torch.zeros(1))
        main = nn.Sequential(
            # Z goes into a linear of size: hid_dim
            nn.Linear(num_chan * input_size * input_size, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, out_dim),

        )
        self.main = main
        self.num_chan = num_chan
        self.input_size = input_size

    def forward(self, inputs):
        inputs = inputs.view(inputs.size(0), -1)
        if isinstance(inputs.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
        else:
            output = self.main(inputs)
        #output = output.mean(0)
        return output


# DCGAN generator
class DCGAN_G(torch.nn.Module):
	def __init__(self, input_size, noise_dim, num_chan, 
                 hid_dim, ngpu=1, selu=True):
		super(DCGAN_G, self).__init__()
		self.__dict__.update(locals())
		self.register_buffer('device_id', torch.zeros(1))
        	main = torch.nn.Sequential()
		# We need to know how many layers we will use at the beginning
		mult = input_size // 8

		### Start block
		# Z_size random numbers
		main.add_module('Start-ConvTranspose2d', torch.nn.ConvTranspose2d(noise_dim, hid_dim * mult, kernel_size=4, stride=1, padding=0, bias=False))
		if selu:
			main.add_module('Start-SELU', torch.nn.SELU(inplace=True))
		else:
			main.add_module('Start-BatchNorm2d', torch.nn.BatchNorm2d(hid_dim * mult))
			main.add_module('Start-ReLU', torch.nn.ReLU())
		# Size = (G_h_size * mult) x 4 x 4

		### Middle block (Done until we reach ? x image_size/2 x image_size/2)
		i = 1
		while mult > 1:
			main.add_module('Middle-ConvTranspose2d [%d]' % i, torch.nn.ConvTranspose2d(hid_dim * mult, hid_dim * (mult//2), kernel_size=4, stride=2, padding=1, bias=False))
			if selu:
				main.add_module('Middle-SELU [%d]' % i, torch.nn.SELU(inplace=True))
			else:
				main.add_module('Middle-BatchNorm2d [%d]' % i, torch.nn.BatchNorm2d(hid_dim * (mult//2)))
				main.add_module('Middle-ReLU [%d]' % i, torch.nn.ReLU())
			# Size = (G_h_size * (mult/(2*i))) x 8 x 8
			mult = mult // 2
			i += 1
        
		### End block
		# Size = G_h_size x image_size/2 x image_size/2
		main.add_module('End-ConvTranspose2d', torch.nn.ConvTranspose2d(hid_dim, num_chan, kernel_size=4, stride=2, padding=1, bias=False))
		main.add_module('End-Tanh', torch.nn.Tanh())
		# Size = n_colors x image_size x image_size
		self.main = main

        	self.apply(weights_init)

	def forward(self, inputs):
    		inputs = inputs.unsqueeze(-1).unsqueeze(-1)
		inputs = inputs.view(inputs.size()[0], inputs.size()[1], 1, 1)
		if isinstance(inputs.data, torch.cuda.FloatTensor) and self.ngpu > 1:
			output = torch.nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
		else:
			output = self.main(inputs)
		return output

# DCGAN discriminator (using somewhat the reverse of the generator)
# Removed Batch Norm we can't backward on the gradients with BatchNorm2d
class DCGAN_D(torch.nn.Module):
	def __init__(self, input_size, num_chan, hid_dim, out_dim = 1, ngpu=1, selu=True):
		super(DCGAN_D, self).__init__()
		self.register_buffer('device_id', torch.zeros(1))
		self.__dict__.update(locals())

		main = torch.nn.Sequential()
		### Start block
		# Size = n_colors x image_size x image_size
		main.add_module('Start-Conv2d', torch.nn.Conv2d(num_chan,  hid_dim, kernel_size=4, stride=2, padding=1, bias=False))
		if selu:
			main.add_module('Start-SELU', torch.nn.SELU(inplace=True))
		else:
			main.add_module('Start-LeakyReLU', torch.nn.LeakyReLU(0.2, inplace=True))
		image_size_new = input_size // 2
		# Size = D_h_size x image_size/2 x image_size/2

		### Middle block (Done until we reach ? x 4 x 4)
		mult = 1
		i = 0
		while image_size_new > 4:
			main.add_module('Middle-Conv2d [%d]' % i, torch.nn.Conv2d(hid_dim * mult, hid_dim * (2*mult), kernel_size=4, stride=2, padding=1, bias=False))
			if selu:
				main.add_module('Middle-SELU [%d]' % i, torch.nn.SELU(inplace=True))
			else:
				main.add_module('Middle-LeakyReLU [%d]' % i, torch.nn.LeakyReLU(0.2, inplace=True))
			# Size = (D_h_size*(2*i)) x image_size/(2*i) x image_size/(2*i)
			image_size_new = image_size_new // 2
			mult = mult*2
			i += 1

		### End block
		# Size = (D_h_size * mult) x 4 x 4
		main.add_module('End-Conv2d', torch.nn.Conv2d(hid_dim * mult, out_dim, kernel_size=4, stride=1, padding=0, bias=False))
		# Note: No more sigmoid in WGAN, we take the mean now
		# Size = 1 x 1 x 1 (Is a real cat or not?)
		self.main = main
        	self.apply(weights_init)
		
	def forward(self, inputs):
		if isinstance(inputs.data, torch.cuda.FloatTensor) and self.ngpu > 1:
			output = torch.nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
		else:
			output = self.main(inputs)
		# From batch_size x 1 x 1 (DCGAN used the sigmoid instead before)
		# Convert from batch_size x 1 x 1 to batch_size
		return output.view(-1)

## Weights init function, DCGAN use 0.02 std
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		# Estimated variance, must be around 1
		m.weight.data.normal_(1.0, 0.02)
		# Estimated mean, must be around 0
		m.bias.data.fill_(0)
