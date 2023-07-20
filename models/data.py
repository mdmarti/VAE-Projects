import torch
from torch.utils.data import Dataset, DataLoader


class toyDataset(Dataset):

	def __init__(self,toyData,sampledt,truedt) -> None:
		
		self.origData = toyData
		skip = int(sampledt/truedt)
		self.data=toyData[::skip]
		self.dt = sampledt
		self.length = self.data.shape[0] - 1
		## needed: slice data by dt? need true dt, ds dt for that
		## should be fine to add though

	def __len__(self):

		return self.length 
	
	def __getitem__(self, index):
		
		single_index = False
		result = []
		try:
			iterator = iter(index)
		except TypeError:
			index = [index]
			single_index = True

		for ii in index:
			result.append((self.data[ii],self.data[ii + 1],self.dt))

		if single_index:
			return result[0]
		return result
						
		
def makeToyDataloaders(ds1,ds2,sampledt,truedt):

	#assert ds1.shape[1] == 3
	ds1 = torch.from_numpy(ds1).type(torch.FloatTensor)
	ds2 = torch.from_numpy(ds2).type(torch.FloatTensor)
	dataset1 = toyDataset(ds1,sampledt,truedt)
	dataset2 = toyDataset(ds2,sampledt,truedt)

	trainDataLoader = DataLoader(dataset1,batch_size=256,shuffle=True,
			      num_workers=4)
	testDataLoader = DataLoader(dataset2,batch_size=256,shuffle=False,
			      num_workers=4)
	
	return {'train':trainDataLoader,'test':testDataLoader}
