import torch
import torch.nn as nn
import torch.nn.functional as F
#from dataset import CustomDataset
#from torch.utils.data import DataLoader

from model import BackgroundSmoother
from tqdm import tqdm
from torch import optim
from torchvision import utils, transforms
from torchvision.transforms import functional as trans_fn


def compute_distribution_loss(images):
	loss = 0
	for i in range(images.shape[0]):
		cur_image = images[i]
		cur_image = cur_image.sort(axis = 1)
		cur_rand = torch.rand(cur_image.values.shape).cuda()
		cur_rand = cur_rand.sort(axis = 1)
		loss += torch.sum(torch.abs(cur_image.values - cur_rand.values))
	loss = loss / images.shape[0]
	return loss

def compute_var_loss(images):
	crop_indices = torch.randint(0, 64 - 16, (images.shape[0], 2))
	variance = 0
	for i in range(images.shape[0]):
		img = images[i, 0 ,crop_indices[i][0]: crop_indices[i][0] + 16, crop_indices[i][1]: crop_indices[i][1] + 16]
		img = img.reshape(-1)
		#print(f'{img.shape} , {torch.max(img)} , {torch.min(img)}')
		variance += (torch.var(img) / (torch.max(img) - torch.min(img)))
	variance = variance / images.shape[0]
	return variance

def compute_eml_loss(images):
	ranges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	eml_buckets = []
	#print(f'images shape = {images.shape}')
	#print(f'grad = {images.grad}')
	prev = torch.zeros(images.shape).cuda()
	for i in range(1, len(ranges)):
		curr = images < ranges[i]
		curr = curr.type(torch.float32)
		diff = curr - prev 
		eml_buckets.append(diff.sum().item())
		prev = curr
	return eml_buckets



def training(epoch, model, optimizer):
	print_img_every = 100
	pbar = tqdm(range(epoch))
	save_model_every = 500

	mean_weight = 1
	var_weight = 1e4
	edge_weight = 1
	eml_weight = 10e-3
	dist_weight = 1e-1
	#pbar = tqdm(range(1))
	for i in pbar:
		optimizer.zero_grad()
		input = torch.randn(128, 64).cuda()

		#labels = torch.randint(10, (128, 1)).cuda()

		output = model(input)

		#mean_loss = compute_mean_loss(labels)
		var_loss = compute_var_loss(output)
		#gaus_loss = compute_gaussian_loss(output)
		#var_loss = 0
		dist_loss = compute_distribution_loss(output)
		eml_buckets = compute_eml_loss(output)
		loss = var_weight * var_loss + dist_weight * dist_loss 
		loss.backward()

		#buckets = Compute_loss(output)
		#print(f'loss = {loss.item()}')
		optimizer.step()
		if((i + 1) % 1 == 0):
			#mean = torch.mean(output)
			msg = f'v: {var_weight * var_loss} | dist: {dist_weight * dist_loss} | eml: {eml_buckets}'
			pbar.set_description(msg)
			
		if((i + 1) % (print_img_every) == 0):
			#print(f'yes')
			with torch.no_grad():
				model.eval()
				gen_inputs = torch.rand(32*3, 64).cuda()
				gen_images = model(input = gen_inputs).data.cpu()
				gen_images = gen_images.reshape(-1, 3, 64, 64)

				utils.save_image(
					gen_images,
					f'./variance_ratio_dist_Shiftrange_{i + 1}.png',
					nrow = 8,
					normalize = True,
					range = (-1, 1)
				)
				
		if((i + 1) % save_model_every == 0):
			torch.save(
				{
				'BackgroundSmoother': model.module.state_dict(),
				'BackgroundSmoother_optimizer': optimizer.state_dict()
				},
				f'./variance_ratio_dist_Shiftrange_{(i + 1)}.model'
			)
	
	
if __name__ == '__main__':
	load_model = 0
	epoch = 1000
	smoother = nn.DataParallel(BackgroundSmoother()).cuda()

	smooth_optimizer = optim.RMSprop(smoother.parameters(), lr = 1e-5)
	
	if(load_model == 1):
		print(f'loading model')
		checkpoint = torch.load('./colourful_letters_sample1/variance_ratio_shiftrange_1000.model')
		smoother.module.load_state_dict(checkpoint['BackgroundSmoother'])
		smooth_optimizer.load_state_dict(checkpoint['BackgroundSmoother_optimizer'])

	training(epoch, smoother, smooth_optimizer)

