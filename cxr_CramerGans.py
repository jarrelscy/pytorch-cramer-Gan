from cramerGan.Models import DCGAN_D as Disc
from cramerGan.Models import DCGAN_G as Gen

from cramerGan.utils import RGBToGray
from cramerGan.cramerGan import train_gans
import numpy as np
import argparse
import os, random
import torch

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
if  __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Gans')    
    parser.add_argument('--weight_decay', type=float, default= 0,
                        help='weight decay for training')
    parser.add_argument('--maxepoch', type=int, default=12800000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default = 0.001, metavar='LR',


                        help='learning rate (default: 0.001)')

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--reuse_weights', action='store_false', default = True,
                        help='continue from last checkout point')
    parser.add_argument('--show_progress', action='store_false', default = True,
                        help='show the training process using images')

    parser.add_argument('--cuda', action='store_false', default=True,
                        help='enables CUDA training')
    
    parser.add_argument('--save_freq', type=int, default= 200, metavar='N',
                        help='how frequent to save the model')
    parser.add_argument('--display_freq', type=int, default= 100, metavar='N',
                        help='plot the results every {} batches')
    
    parser.add_argument('--data_root', type=str, required=True, metavar='N',
                        help='path to dataset.')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size.')

    parser.add_argument('--image_size', type=int, default=64, metavar='N',
                        help='image size.')
    parser.add_argument('--random_crop', type=int, default=8, metavar='N',
                        help='random crop size.')
    parser.add_argument('--gpunum'  , type=int, default=0, help='which of GPUs to use')

    parser.add_argument('--gp_lambda', type=int, default=10, metavar='N',
                        help='the lambda parameter.')

    parser.add_argument('--noise_dim', type=int, default=10, metavar='N',
                        help='dimension of gaussian noise.')
    parser.add_argument('--workers', type=int, default=10, metavar='N',
                        help='number of data workers.')
    parser.add_argument('--median_filter_length', type=int, default=100, metavar='N',
                        help='number of losses to put median filter over.')
    parser.add_argument('--ncritic', type=int, default= 5, metavar='N',
                        help='the number of times to train the critic per batch.')

    parser.add_argument('--save_folder', type=str, default= 'tmp_images', metavar='N',
                        help='folder to save the temper images.')

    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()
    os.system('mkdir {0}'.format(args.save_folder))
    torch.cuda.set_device(args.gpunum)
    args.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    cudnn.benchmark = True
    
    
    netD = Disc(input_size = args.image_size, num_chan =1, hid_dim = 64, out_dim = 16 )
    netG = Gen(input_size  = args.image_size, noise_dim = args.noise_dim, num_chan=1, hid_dim= 64)

    if args.cuda:
        netD = netD.cuda()
        netG = netG.cuda()

    dataset = dset.ImageFolder(root=args.data_root,
                               transform=transforms.Compose([
                                  transforms.Scale(args.image_size),
                                   transforms.RandomCrop(args.image_size, args.random_crop),
                                   transforms.ToTensor(),
                                   RGBToGray(),
                                   transforms.Normalize((0.5,), (0.5,)), 
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=int(args.workers))
    data_sampler = dataloader
    model_root, model_name = 'model', 'cxr_CramerWgan'

    
    train_gans(data_sampler, model_root, model_name, netG, netD,args)
