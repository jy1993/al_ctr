from torch.utils.data import dataset
import pandas as pd


class Ali(dataset):
	"""docstring for Ali"""
	def __init__(self, arg):
		super(Ali, self).__init__()
		self.arg = arg
		if self.arg.test_flag:
			d_path = self.arg.data_path + '/test'	
		elif self.arg.train_flag:
			d_path = self.arg.data_path + '/train'		
		else: 
			d_path = self.arg.data_path + '/val'
		self.cdata = open(f'{d_path}').read().split('\n')

	def __getitem__(self, index):
		data_and_label = self.cdata[index].split(None)
		data, label = data_and_label[:-1], data_and_label[-1]
		return data, label

	def __len__(self):
		return len(self.cdata)		
