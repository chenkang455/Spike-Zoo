import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import shutil
import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import warnings
warnings.filterwarnings('ignore')

import config      as cfg
import structure
import netloss
from load_data import *
import time
from utils import RawToSpike, cal_para

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     # random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(20)


def initialize():
	"""
	# clear some dir if necessary
	make some dir if necessary
	make sure training from scratch
	:return:
	"""
	##
	if not os.path.exists(cfg.model_name):
		os.mkdir(cfg.model_name)

	if not os.path.exists(cfg.debug_dir):
		os.mkdir(cfg.debug_dir)

	if not os.path.exists(cfg.log_dir):
		os.mkdir(cfg.log_dir)


def train(in_data, gt_raw_data, q, nd, nl, model, loss, device, optimizer):
	l1loss_list = []
	l1loss_total = 0
	for time_ind in range(cfg.frame_num):
		ft1 = in_data[:, time_ind:time_ind+1, :, :] 	 		# the t-th input frame
		fgt = gt_raw_data[:, 0:1, :, :] 						# the t-th gt frame
		if time_ind == 0:
			# ft0_fusion = ft1
			input = ft1
		else:
			# ft0_fusion = ft0_fusion_data							 # the t-1 fusion frame
			input = torch.cat([ft1, ft0_fusion_data], dim=1)
		model.train()
		fpn_denoise, img_true, fusion_out, denoise_out, refine_out, ft_denoise_out_d0, fgt_d0 = model(input, fgt, q, nd, nl)
		fgt = F.pad(fgt, [0, 0, 3, 3], mode="reflect")
		loss_ft = 0
		for i in range(4):
			loss_ft += loss(ft_denoise_out_d0[:, i:i+1, :, :], fgt_d0[:, i:i+1, :, :])
		l1loss = loss(refine_out, fgt) + loss(fusion_out, fgt) + loss(denoise_out, fgt) + loss_ft

		l1loss_list.append(l1loss)
		l1loss_total += l1loss

		ft0_fusion_data = fusion_out[:, :, 3:253, :]

	total_loss = l1loss_total / (cfg.frame_num)

	optimizer.zero_grad()
	total_loss.backward()
	optimizer.step()

	del in_data, gt_raw_data
	return total_loss.item(), loss_ft.item()


def evaluate(model, q, nd, nl, psnr, writer, iter, val_loader):
	print('Evaluate...')
	cnt = 0
	total_psnr = 0
	total_psnr_traditional = 0
	total_psnr_fusion = 0
	total_psnr_denoise = 0
	total_psnr_refine = 0
	model.eval()
	with torch.no_grad():
		for in_data, gt_raw_data in val_loader:
			in_data = in_data.to(cfg.device)
			gt_raw_data = gt_raw_data.to(cfg.device)
			frame_psnr = 0
			frame_psnr_traditional = 0
			frame_psnr_fusion = 0
			frame_psnr_denoise = 0
			frame_psnr_refine = 0
			for time_ind in range(cfg.frame_num):
				ft1 = in_data[:, time_ind: (time_ind + 1), :, :]
				fgt = gt_raw_data[:, 0:1, :, :]
				if time_ind == 0:
					# ft0_fusion = ft1
					input = ft1
				else:
					# ft0_fusion = ft0_fusion_data
					input = torch.cat([ft1, ft0_fusion_data], dim=1)

				# input = torch.cat([ft0_fusion, ft1], dim=1)

				fpn_denoise, img_true, fusion_out, denoise_out, refine_out, ft_denoise_out_d0, fgt_d0 = model(input, fgt, q, nd, nl)
				fgt = F.pad(fgt, [0, 0, 3, 3], mode="reflect")
				ft0_fusion_data = fusion_out[:, :, 3:253, :]

				frame_psnr += psnr(fpn_denoise[:, :, 3:253, :], fgt[:, :, 3:253, :])
				frame_psnr_traditional += psnr(img_true, fgt[:, :, 3:253, :])
				frame_psnr_fusion += psnr(fusion_out[:, :, 3:253, :], fgt[:, :, 3:253, :])
				frame_psnr_denoise += psnr(denoise_out[:, :, 3:253, :], fgt[:, :, 3:253, :])
				frame_psnr_refine += psnr(refine_out[:, :, 3:253, :], fgt[:, :, 3:253, :])
			cv2.imwrite("./result/fpn_denoise.png", np.transpose(fpn_denoise[:, :, 3:253, :].detach().cpu().numpy()[0], [1, 2, 0]) * 255.0)
			cv2.imwrite("./result/img_true.png", np.transpose(img_true.detach().cpu().numpy()[0], [1, 2, 0]) * 255.0)
			cv2.imwrite("./result/fusion_out.png", np.transpose(fusion_out[:, :, 3:253, :].detach().cpu().numpy()[0], [1, 2, 0]) * 255.0)
			cv2.imwrite("./result/input.png", np.transpose(ft1[:, :, 3:253, :].detach().cpu().numpy()[0], [1, 2, 0]) * 255.0)
			cv2.imwrite("./result/gt.png", np.transpose(fgt[:, :, 3:253, :].detach().cpu().numpy()[0], [1, 2, 0]) * 255.0)
			cv2.imwrite("./result/denoise_out.png",  np.transpose(denoise_out[:, :, 3:253, :].detach().cpu().numpy()[0], [1, 2, 0]) * 255.0)
			cv2.imwrite("./result/refine_out.png", np.transpose(refine_out[:, :, 3:253, :].detach().cpu().numpy()[0], [1, 2, 0]) * 255.0)
			frame_psnr = frame_psnr / cfg.frame_num
			frame_psnr_traditional = frame_psnr_traditional / cfg.frame_num
			frame_psnr_fusion = frame_psnr_fusion / cfg.frame_num
			frame_psnr_denoise = frame_psnr_denoise / cfg.frame_num
			frame_psnr_refine = frame_psnr_refine / cfg.frame_num
			total_psnr += frame_psnr
			total_psnr_traditional += frame_psnr_traditional
			total_psnr_fusion += frame_psnr_fusion
			total_psnr_denoise += frame_psnr_denoise
			total_psnr_refine += frame_psnr_refine
			cnt += 1
			del in_data, gt_raw_data
		total_psnr = total_psnr / cnt
		total_psnr_traditional = total_psnr_traditional / cnt
		total_psnr_fusion = total_psnr_fusion / cnt
		total_psnr_denoise = total_psnr_denoise / cnt
		total_psnr_refine = total_psnr_refine / cnt
	print('Eval_Total_PSNR_fpn_denoise              |   ', ('%.8f' % total_psnr.item()))
	print('Eval_Total_PSNR_traditional             |   ', ('%.8f' % total_psnr_traditional.item()))
	print('Eval_Total_PSNR_fusion             |   ', ('%.8f' % total_psnr_fusion.item()))
	print('Eval_Total_PSNR_denoise             |   ', ('%.8f' % total_psnr_denoise.item()))
	print('Eval_Total_PSNR_refine             |   ', ('%.8f' % total_psnr_refine.item()))

	writer.add_scalar('PSNR', total_psnr.item(), iter)
	torch.cuda.empty_cache()
	return total_psnr, total_psnr_traditional, total_psnr_fusion, total_psnr_denoise, total_psnr_refine


def main():
	"""
	Train, Valid, Write Log, Write Predict ,etc
	:return:
	"""
	checkpoint = cfg.checkpoint
	start_epoch = cfg.start_epoch
	start_iter = cfg.start_iter
	best_psnr = 0

	## use gpu
	device = cfg.device
	ngpu = cfg.ngpu
	cudnn.benchmark = True

	## tensorboard --logdir runs
	writer = SummaryWriter(cfg.log_dir)

	## initialize model
	model = structure.MainDenoise(mode="train")

	model = model.to(device)
	# model_ill = model_ill.to(device)
	loss = netloss.L1Loss().to(device)
	psnr = netloss.PSNR().to(device)

	learning_rate = cfg.learning_rate
	optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

	## load pretrained part model and fix it
	pre_dict = torch.load("./model/nim.pth")['model']
	model_dict = model.state_dict()
	model_dict.update(pre_dict)
	model.load_state_dict(model_dict)
	for i, p in enumerate(model.parameters()):
		# print(p)
		if i < 19:
			p.requires_grad = False
	for name, p in model.named_parameters():
		print(f'{name}:\t{p.requires_grad}')
	## load pretrained model
	if checkpoint is not None:
		print('--- Loading Pretrained Model ---')
		checkpoint = torch.load(checkpoint)
		start_epoch = checkpoint['epoch']
		start_iter = checkpoint['iter']
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
	iter = start_iter

	if torch.cuda.is_available() and ngpu > 1:
		model = nn.DataParallel(model, device_ids=list(range(ngpu)))


	Q, Nd, Nl = cal_para()
	train_data_name_queue = generate_file_list("train")
	train_dataset = loadImgs(train_data_name_queue, "train", use_cache=False)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=0, shuffle=True, pin_memory=True)

	## load dataset to memory
	for train_data, train_label in tqdm.tqdm(train_loader, position=0, desc='load train dataset'):
		pass
	train_loader.dataset.set_use_cache(use_cache=True)
	train_loader.num_workers = 2

	eval_data_name_queue = generate_file_list("test")
	eval_dataset = loadImgs(eval_data_name_queue, "test", use_cache=False)
	eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True)
	for test_data, test_label in tqdm.tqdm(eval_loader, position=0, desc='load test dataset'):
		pass
	eval_loader.dataset.set_use_cache(use_cache=True)
	eval_loader.num_workers = 2

	temp = input("input anything to continue!")

	q = torch.from_numpy(Q).to(device)
	nd = torch.from_numpy(Nd).to(device)
	nl = torch.from_numpy(Nl).to(device)

	for epoch in range(start_epoch, cfg.epoch):
		with tqdm.tqdm(total=np.ceil(len(train_loader))) as t:
			mean_train_loss = 0
			mean_grad_loss = 0
			mean_ft_loss = 0
			for i, (input_, label) in enumerate(train_loader):
				t.set_description('Epoch %i' % epoch)
				in_data = input_.to(device)
				gt_raw_data = label.to(device)
				train_loss, ft_loss = train(in_data, gt_raw_data, q, nd, nl, model, loss, device, optimizer)
				mean_train_loss += train_loss
				mean_ft_loss += ft_loss
				t.set_postfix(train_loss=mean_train_loss / (i+1), grad_loss=mean_grad_loss / (i+1), ft_loss=mean_ft_loss / (i+1))
				t.update(1)
				iter = iter + 1
		if epoch % cfg.save_epoch == 0:
			torch.save({
				'epoch': epoch,
				'iter': iter,
				'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'best_psnr': best_psnr},
				cfg.model_save_root)

		if epoch % cfg.eval_epoch == 0:
			print(epoch)
			eval_psnr, eval_psnr_traditional, eval_psnr_fusion, eval_psnr_denoise, eval_psnr_refine = evaluate(model, q, nd, nl, psnr, writer, iter, eval_loader)
			if eval_psnr_refine > best_psnr:
				best_psnr = eval_psnr_refine
				torch.save({
					'epoch': epoch,
					'iter': iter,
					'model': model.state_dict(),
					'optimizer': optimizer.state_dict(),
					'best_psnr': best_psnr},
					cfg.best_model_save_root)
	writer.close()


if __name__ == '__main__':
	initialize()
	main()
