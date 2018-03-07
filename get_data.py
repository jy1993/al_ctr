from torch.utils.data import dataset


class Ali(dataset):
	"""docstring for Ali"""
	def __init__(self, arg):
		super(Ali, self).__init__()
		self.arg = arg

	def __getitem__(self, index):
		pass

	def __len__(self):
		pass
		