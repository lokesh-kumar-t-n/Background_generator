import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.nn.functional as F
from cgan_letters_sample.model import generator, discriminator
from tqdm import tqdm
from torch import optim
from torchvision import utils, transforms
from torchvision.transforms import functional as trans_fn
import imageio

from colourful_letters_sample1.variance_ratio_dist_Shiftrange_model import BackgroundSmoother

def edge_detector(image):
	edge_filter = torch.tensor([[1., 1., 1.],[1., -8., 1.], [1., 1., 1.]])
	edge_filter.unsqueeze_(0)
	edge_filter.unsqueeze_(0)
	image.unsqueeze_(0)
	edge = F.conv2d(image, edge_filter, padding = 1)
	print(f'edge = {edge.shape}')
	return edge

def add_as_channel(images, background_images):
	for i in range(32):
		background_images[i][torch.randint(3, (1,1))] = background_images[i][0].transpose(0, 1)
		background_images[i][torch.randint(3, (1,1))] = background_images[i][1].transpose(0, 1).flip(1)

		background_images[i][torch.randint(3, (1,1))] = images[i][0]

	utils.save_image(
		background_images,
		f'colourful_channel.jpg',
		nrow = 8,
		normalize = True,
		range = (-1, 1)
	)

def add_as_edge(images, background_images):
	for i in range(32):
		background_images[i][torch.randint(3, (1,1))] = background_images[i][0].transpose(0, 1)
		background_images[i][torch.randint(3, (1,1))] = background_images[i][1].transpose(0, 1).flip(1)

		background_images[i][torch.randint(3, (1,1))] += edge_detector(images[i])
	utils.save_image(
		background_images,
		f'colourful_edge1.jpg',
		nrow = 8,
		normalize = True,
		range = (-1, 1)
	)



def letter_colour(images, background_images, threshold):
	for i in range(32):
		background_images[i][torch.randint(3, (1,1))] = background_images[i][0].transpose(0, 1)
		background_images[i][torch.randint(3, (1,1))] = background_images[i][1].transpose(0, 1).flip(1)

		background_images[i] = background_images[i] * (images[i] > threshold)

	utils.save_image(
		background_images,
		f'colourful_letter.jpg',
		nrow = 8,
		normalize = True,
		range = (-1, 1)
	)

def background_colour(images, background_images, threshold):
	for i in range(32):
		background_images[i][torch.randint(3, (1,1))] = background_images[i][0].transpose(0, 1)
		background_images[i][torch.randint(3, (1,1))] = background_images[i][1].transpose(0, 1).flip(1)

		background_images[i] = background_images[i] * (images[i] < threshold)
	utils.save_image(
		background_images,
		f'colourful_background.jpg',
		nrow = 8,
		normalize = True,
		range = (-1, 1)
	)
if __name__ == '__main__':
	load_model = 1
	gen = nn.DataParallel(generator()).cuda()
	disc = nn.DataParallel(discriminator()).cuda()

	loss = nn.BCELoss()

	gen_optimizer = optim.Adam(gen.parameters(), lr = 0.0002, betas = (0.5, 0.999))
	disc_optimizer = optim.Adam(disc.parameters(), lr = 0.0002, betas = (0.5, 0.999))

	#my_data = CustomDataset()
	#print(f'batch_size = {batch_size}')
	#dataloader = DataLoader(my_data, batch_size = batch_size, shuffle = True)
	#print(f'len = {len(dataloader)}')
	if(load_model == 1):
		print(f'loaded model')
		checkpoint = torch.load('./cgan_letters_sample/emnist_bceloss_model_200.model')
		gen.module.load_state_dict(checkpoint['gen'])
		disc.module.load_state_dict(checkpoint['disc'])
		gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
		disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])
	
	smoother = nn.DataParallel(BackgroundSmoother()).cuda()

	if(load_model == 1):
		checkpoint = torch.load('./colourful_letters_sample1/variance_ratio_dist_Shiftrange_1000.model')
		smoother.module.load_state_dict(checkpoint['BackgroundSmoother'])
	
	with torch.no_grad():
		
		img_labels = torch.LongTensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,1,2,3,4,5,6]).cuda()
		gen_inputs = torch.randn(32, 64).cuda()
		gen_images = gen(input = gen_inputs, labels = (img_labels - 1)).data.cpu()
		print(f'gen_images = {gen_images.shape}')

		gen_inputs = torch.randn(32*3, 64).cuda()
		background_images = smoother(gen_inputs).data.cpu()
		background_images = background_images.reshape(32, 3, 64, 64)
		

		background_colour(gen_images, background_images, 0.35)
		#print(f' so')