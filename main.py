import torch
from deepfm import DeepFM
from get_data import Ali
from torch.utils.data import dataloader
from torch.optim import Adam
from torch.autograd import Variable
import os


class Config():
	# data
	data_path = '../data'
	train_flag = True
	test_flag = False

	# optimizier
	beta1 = 0.5
	beta2 = 0.999

	# training
	epochs = 10
	batch_size = 64

	# visualize and save
	eval_every = 1
	model_path = '../save/models'
	output_path = '../save/output'

def to_var(x, volatile=False):
	x = Variable(x)
	if torch.cuda.is_available():
		x = x.cuda()
	return x

config = Config()

def train(**kwargs):
	for k_, v_ in kwargs:
		setattr(config, k_, v_)
	train_dataset, val_dataset, test_dataset = Ali(config)
	train_loader = dataloader(dataset=train_dataset, batch_size=config.batch_size, 
		shuffle=True, drop_last=True)
	val_loader = dataloader(dataset=val_dataset, batch_size=config.batch_size)

	model = DeepFM()

	# testing 
	if config.test_flag:
		test_loader = dataloader(dataset=test_dataset, batch_size=config.batch_size)
		model.load_state_dict(torch.load(os.path.join(config.model_path, '_best')))
		test(model, test_loader, config.output_path)

	criterion = torch.nn.BCELoss()
	optimizer = Adam(model.parameters, lr=config.lr, betas=(config.beta1, config.beta2))
	best_val_loss = 1e6
	if torch.cuda.is_available():
		model.cuda()
		criterion.cuda()
	
	# resume training
	start = 0
	if config.resume:
		model_epoch = [int(fname.split('_')[-1]) for fname in os.listdir(config.model_path) 
			if 'best' not in fname]
		start = max(model_epoch)
		model.load_state_dict(torch.load(os.path.join(config.model_path, '_epoch_{start}')))
	if start >= config.epochs:
		print('Training already Done!')
		return 

	for i in range(start, config.epochs):
		for ii, (c_data, labels) in enumerate(train_loader):
			c_data = to_var(c_data)
			labels = to_var(labels)

			pred = model(c_data)
			loss = criterion(pred, labels, criterion)
			loss.backward()
			optimizer.step()

		if (ii + 1) % config.eval_every == 0:
			val_loss = val(model, val_loader)
			print(f'''epochs: {i + 1}/{config.epochs} batch: {ii + 1}/{len(train_loader)}\t
						train_loss: {loss.data[0] / c_data.size(0)}, val_loss: {val_loss}''')

			torch.save(model.state_dict(), os.path.join(config.model_path, '_epoch_{i}'))
			if val_loss < best_val_loss:
				torch.save(model.state_dict(), os.path.join(config.model_path, '_best'))

def val(fmodel, data_loader, loss_f):
	fmodel.eval()
	loss = 0
	for (c_data, labels) in data_loader:
		c_data, labels = to_var(c_data), to_var(labels)
		pred = fmodel(c_data)
		loss += loss_f(pred, labels) / c_data.size(0)
	fmodel.train()
	return loss.data[0]

def test(fmodel, data_loader, output_path):
	fmodel.eval()
	preds = []
	for (c_data, _) in data_loader:
		c_data = to_var(c_data, volatile=True)
		preds.append(fmodel(c_data))
	print(preds)

if __name__ == '__main__':
	from fire import Fire
	Fire()