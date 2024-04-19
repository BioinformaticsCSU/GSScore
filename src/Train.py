
import shutil
import sys, os, argparse, math
import traceback

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, Tensor
import torch.nn.functional as F
import h5py
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.distributed as dist

from model import *

# Define GetVoxelData class
class GetGraphData(Dataset):
	def __init__(self, SHELLs, GraphFile, ListFile, max_hop=8
				 , bCheckDataset=True, bUseEdgeFeature=True, bPreload=False, bUseDegree=False):
		self.SHELLs = SHELLs
		self.GraphFile = GraphFile
		self.HDF5 = None
		self.ListFile = ListFile
		self.max_hop = max_hop
		self.PDBNames = np.loadtxt(ListFile, skiprows=0, usecols=[0], dtype=str)
		self.PoseNames = np.loadtxt(ListFile, skiprows=0, usecols=[1], dtype=str)
		self.ExpDatas = np.loadtxt(ListFile, skiprows=0, usecols=[2], dtype=float)
		self.bUseEdgeFeature = bUseEdgeFeature
		self.PreloadData = {}
		self.bPreload = bPreload
		self.bUseDegree=bUseDegree
		# self.HDF5 = h5py.File(self.GraphFile, 'r')
		if bCheckDataset:
			self.Func_CheckDataset()

	def Func_CheckDataset(self, maxExpData=10.0):
		print('Checking data integrity ...')
		self.HDF5 = h5py.File(self.GraphFile, 'r')
		newPDBNames = []
		newPoseNames = []
		newExpDatas = []
		total = len(self.PoseNames)
		for index in range(len(self.PoseNames)):
			if self.ExpDatas[index] > maxExpData:
				continue
			PDBName = self.PDBNames[index]
			PoseName = self.PoseNames[index]
			if PoseName.strip() == 'Lig_native' or PoseName.strip() == 'Lig_0':
				if ''.join(['/', PDBName, '/Lig_native']) in self.HDF5:
					MyDebug = ''.join(['/', PDBName, '/Lig_native'])
					PoseName = 'Lig_native'
				else:
					MyDebug = ''.join(['/', PDBName, '/Lig_0'])
					PoseName = 'Lig_0'
			else:
				MyDebug = ''.join(['/', PDBName, '/', PoseName])
			if MyDebug in self.HDF5:
				TempGroup = self.HDF5.require_group(MyDebug)
				if ("LigandAtomFeatures" in TempGroup):
					newPDBNames.append(PDBName)
					newPoseNames.append(PoseName)
					newExpDatas.append(self.ExpDatas[index])
			if index % 10000 == 0:
				print('\r%d/%d ...' % (index+1, total), end='', flush=True)
		self.PDBNames = np.array(newPDBNames)
		self.PoseNames = np.array(newPoseNames)
		self.ExpDatas = np.array(newExpDatas)
		self.HDF5.close()
		self.HDF5 = None
		f = open('TrainCurrentDecoys.list', 'w')
		for i in range(self.PDBNames.shape[0]):
			f.write('%-16s%-16s%f\n' % (self.PDBNames[i], self.PoseNames[i], self.ExpDatas[i]))
		f.close()
		print('Finished checking!')

	def Func_GetPDBList(self):
		return self.PDBNames, self.PoseNames, self.ExpDatas

	def __getitem__(self, index):
		PDBName = self.PDBNames[index]
		PoseName = self.PoseNames[index]
		ExpData = self.ExpDatas[index].reshape(-1).astype(np.float32)
		# Read VoxelDataSet file data
		# print('subprocess %d get index=%d' % (os.getpid(), index))
		if self.HDF5 is None and self.bPreload == False:
			self.HDF5 = h5py.File(self.GraphFile, 'r')

		try:
			if PoseName.strip() == 'Lig_native' or PoseName.strip() == 'Lig_0':
				if ''.join(['/', PDBName, '/Lig_native']) in self.HDF5:
					MyDebug = ''.join(['/', PDBName, '/Lig_native'])
				else:
					MyDebug = ''.join(['/', PDBName, '/Lig_0'])
			else:
				MyDebug = ''.join(['/', PDBName, '/', PoseName])
			Group = self.HDF5.require_group(MyDebug)

			LigandAtomFeatures = torch.LongTensor(Group["LigandAtomFeatures"][:])
			num_atoms = LigandAtomFeatures.shape[0]
			offset_list = [0, 8, 10, 6, 6, 9, 7, 10]
			tempLigandAtomFeatures = torch.zeros([num_atoms, sum(offset_list)+2], dtype=torch.int8)
			sum_k = 0
			for index, k in enumerate(offset_list):
				sum_k += k
				tempLigandAtomFeatures.scatter_(-1, LigandAtomFeatures[:, index:index+1]+sum_k, 1)
			LigandAtomFeatures = tempLigandAtomFeatures

			attn_bias_list = []
			spatial_pos_list = []
			x_list = []
			edge_input_list = []
			degree_list = []
			NumAtomOfEachShell = []
			AtomFeatureSize = 0
			hop_max = 0
			edge_size = 0
			# decoy_list = []
			for shell in range(1, self.SHELLs+1):
				TempGroup = Group.require_group(MyDebug+'/'+str(shell))
				ProteinAtomFeatures = torch.LongTensor(TempGroup["ProteinAtomFeatures"][:])
				if ProteinAtomFeatures.shape[0] <= 0:
					# return None, None, None, None, None, PDBName+' '+PoseName+' '+str(ExpData), ExpData
					attn_bias_list.append(None)
					spatial_pos_list.append(None)
					x_list.append(None)
					edge_input_list.append(None)
					degree_list.append(None)
					NumAtomOfEachShell.append(0)
					continue
				offset_list = [0, 8, 10, 6, 6, 9, 22]
				num_atoms = ProteinAtomFeatures.shape[0]
				tempProteinAtomFeatures = torch.zeros([num_atoms, sum(offset_list)+37], dtype=torch.int8)
				sum_k = 0
				for i,k in enumerate(offset_list):
					sum_k += k
					tempProteinAtomFeatures.scatter_(-1, ProteinAtomFeatures[:, i:i+1]+sum_k, 1)
				# check = (tempProteinAtomFeatures==tempProteinAtomFeatures1).all()
				ProteinAtomFeatures = tempProteinAtomFeatures
				AdjacencyMatrix = torch.as_tensor(TempGroup["AdjacencyMatrix"][:], dtype=torch.float32)
				AdjacencyDistMatrix = torch.as_tensor(TempGroup["AdjacencyDistMatrix"][:], dtype=torch.float32)
				if (AdjacencyMatrix is not None) and self.bUseDegree:
					t = (AdjacencyMatrix>0.0)*(AdjacencyMatrix<=6.0)*1
					degree = t.sum(axis=-1)
				else:
					degree = torch.zeros(AdjacencyDistMatrix.shape[0])

				num_atom = AdjacencyDistMatrix.shape[0]
				attn_bias = torch.zeros([num_atom + 1, num_atom + 1])
				spatial_pos = torch.ones_like(AdjacencyDistMatrix, dtype=torch.float32)
				spatial_pos = spatial_pos * AdjacencyDistMatrix
				# spatial_pos = AdjacencyDistMatrix.copy()
				for i in range(spatial_pos.shape[0]):
					spatial_pos[i, i] = 1.0
				AtomFeatureSize = ProteinAtomFeatures.shape[1] + LigandAtomFeatures.shape[1]
				x = torch.zeros([num_atom, AtomFeatureSize], dtype=torch.int8)
				x[:ProteinAtomFeatures.shape[0], :ProteinAtomFeatures.shape[1]] = ProteinAtomFeatures
				x[ProteinAtomFeatures.shape[0]:, ProteinAtomFeatures.shape[1]:] = LigandAtomFeatures

				edge_input = torch.LongTensor(TempGroup["edge_input"][:])
				num_atom = edge_input.shape[0]
				hop_max = edge_input.shape[2]
				tempedge_input = torch.zeros([num_atom, num_atom, hop_max, 8], dtype=torch.int8)
				tempedge_input.scatter_(-1, edge_input.unsqueeze(-1), 1)
				tempedge_input[:, :, :, 0] = 0
				edge_input = tempedge_input
				edge_size = edge_input.shape[-1]
				if not self.bUseEdgeFeature:
					edge_input *= 0

				attn_bias_list.append(attn_bias)
				spatial_pos_list.append(spatial_pos)
				x_list.append(x)
				edge_input_list.append(edge_input)
				degree_list.append(degree)
				NumAtomOfEachShell.append(num_atom)
				# decoy_list.append(PDBName+' '+PoseName+' '+str(ExpData))

			return attn_bias_list, spatial_pos_list, x_list, edge_input_list, degree_list\
				, [AtomFeatureSize, hop_max, edge_size], NumAtomOfEachShell\
				, PDBName+' '+PoseName+' '+str(ExpData), ExpData
		except Exception as err:
			print('ERROR in %s' % (MyDebug))
			print(err)
			traceback.print_exc()
			sys.exit()

	def __len__(self):
		return len(self.ExpDatas)

# callback for __getitem__
def Func_DatasetPostProcessing(batch_data):
	SHELLs = 0
	AtomFeatureSize = 0
	max_hop = 0
	edge_size = 0
	for decoy in batch_data:
		if decoy[0] is not None:
			SHELLs = len(decoy[0])
			AtomFeatureSize, max_hop, edge_size = decoy[5]
			break
	if SHELLs == 0:
		return None
	else:
		return_data = {}
		batch_size = len(batch_data)
		ExpData = torch.ones([batch_size, 1])
		sample_ids = []
		NumAtomOfEachShell = []
		for index, decoy in enumerate(batch_data):
			NumAtomOfEachShell.append(decoy[6])
			ExpData[index] = torch.Tensor(decoy[-1])
			sample_ids.append(decoy[-2])
		NumAtomOfEachShell = np.array(NumAtomOfEachShell)
		MaxNumAtomOfEachShell = np.max(NumAtomOfEachShell, axis=0)
		return_data['ExpData'] = ExpData
		for shell in range(SHELLs):
			return_data[shell+1] = {}
			Num_bias = MaxNumAtomOfEachShell[shell] + 1
			Num_Atom = MaxNumAtomOfEachShell[shell]
			attn_bias = torch.ones([batch_size, Num_bias, Num_bias]) * (float('-inf'))
			spatial_pos = torch.zeros([batch_size, Num_Atom, Num_Atom])
			x = torch.zeros([batch_size, Num_Atom, AtomFeatureSize])
			edge_input = torch.zeros([batch_size, Num_Atom, Num_Atom, max_hop, edge_size])
			in_degree = torch.zeros([batch_size, Num_Atom])
			out_degree = torch.zeros([batch_size, Num_Atom])
			hasGraph = torch.zeros([batch_size], dtype=torch.int8)
			listOfAvailable = []
			for index, decoy in enumerate(batch_data):
				# tempX, tempY = batch[0].shape[:2]
				check = False
				if decoy[0][shell] is not None:
					tempX, tempY = decoy[0][shell].shape[:2]
					attn_bias[index][:, :tempY] = 0#decoy[0][shell]
					check = True
					if (attn_bias[index]==float('-inf')).all():
						check = False
				if decoy[1][shell] is not None:
					tempX, tempY = decoy[1][shell].shape[:2]
					spatial_pos[index][:tempX, :tempY] = decoy[1][shell]
					check = True
					if (spatial_pos[index]==float('inf')).any():
						check = False
				if decoy[2][shell] is not None:
					tempX, tempY = decoy[2][shell].shape[:2]
					x[index][:tempX, :tempY] = decoy[2][shell]
					check = True
				if decoy[3][shell] is not None:
					tempX, tempY = decoy[3][shell].shape[:2]
					edge_input[index][:tempX, :tempY] = decoy[3][shell]
					in_degree[index][:tempX] = decoy[4][shell]
					out_degree[index][:tempX] = decoy[4][shell]
					check = True
				if check:
					hasGraph[index] = 1
					listOfAvailable.append(index)

			return_data[shell+1]['attn_bias'] = attn_bias[listOfAvailable]
			return_data[shell+1]['spatial_pos'] = spatial_pos[listOfAvailable]
			return_data[shell+1]['x'] = x[listOfAvailable]
			return_data[shell+1]['edge_input'] = edge_input[listOfAvailable]
			return_data[shell+1]['in_degree'] = in_degree[listOfAvailable]
			return_data[shell+1]['out_degree'] = out_degree[listOfAvailable]
			return_data[shell+1]['hasGraph'] = hasGraph

	return return_data,  sample_ids # , ExpData_return


# reload multi GPU model
def load_MultiGPUS(model, model_path, kwargs):
	state_dict = torch.load(model_path, **kwargs)
	# create new OrderedDict that does not contain `module.`
	from collections import OrderedDict
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = k[7:]  # remove `module.`
		new_state_dict[name] = v
	# load params
	model.load_state_dict(new_state_dict)
	return model.cuda()


def argparser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-F', type=str, required=True, help='train set (*.hdf5)')
	parser.add_argument('--UseTmpfs', type=int, default=0
						, help='Use /dev/shm/ as tmpfs folder to store .hdf file? 0=No (default);1=Yes')
	parser.add_argument('-L', type=str, required=True, help='List of train pose (PDB    Lig_*    RMSD)')
	parser.add_argument('-S', type=int, default=0, help='start Epoch (default: 0)')
	parser.add_argument('--net', type=str, default=''
						, help='start parameters (default: ''; Net_Epoch-k.pt means reload parameters from k-th Epoch)')
	parser.add_argument('--shell', type=int, default=10
						,
						help='number of shells to be used in each decoy, (default: 10)')
	parser.add_argument('--emb_dim', type=int, default=128
						,
						help='dimension of atom embedding vector, (default: 128)')
	parser.add_argument('--ffn_emb', type=int, default=128
						,
						help='dimension of ffn embedding, (default: 128)')
	parser.add_argument('--mlp', type=int, default=1
						,
						help='number of final mlp layers, (default: 1)')
	parser.add_argument('--heads', type=int, default=8, help='number of heads in attention (default: 8)')
	parser.add_argument('--multi_hop_max_dist', type=int, default=8, help='number of hop_max_dist (default: 8)')
	parser.add_argument('--dropout', type=float, default=0.0, help='dropout in Fairseq (default: 0.0)')
	parser.add_argument('--num_epoch', type=int, default=150, help='number of Epoch (default: 150)')
	parser.add_argument('--lr', type=float, default=0.0005, help='Learning Rate(default: 0.0005)')
	parser.add_argument('--wd', type=float, default=0.0001, help='weight decay for optimizer(default: 0.0001)')
	parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
	parser.add_argument('--plotgap', type=int, default=100, help='plot train loss per iterations (default: 100)')
	parser.add_argument('--num_producer_threads', type=int, default=4
						, help='number of producer threads (default: 4)')
	parser.add_argument('--num_encoder_layers', type=int, default=16, help='num of encoder (default: 16)')
	parser.add_argument('--gpu', type=str, default='0',
						help='Choose gpu id, example: \'0,1\'(specify use gpu 0, 1 or any other)')
	parser.add_argument('--loss', type=int, default=0,
						help='0=MSE(default); 1=MAE')
	parser.add_argument('--UseEdgeFeature', type=int, default=1,
						help='0=do NOT use edge features in Graphormer; 1=Yes!Use it!(default)')
	parser.add_argument('--UseDegree', type=int, default=0,
						help='0=do NOT use degree in Graphormer(default); 1=Yes!Use it!')
	parser.add_argument('--CheckData', type=int, default=1,
						help='0=do NOT data integrity; 1=Yes!Check it!(default)')
	parser.add_argument('--opt', type=int, default=1,
						help='0=Adam; 1=AdamW(default)')
	parser.add_argument('--preload', type=int, default=0,
						help='0=Not pre-load data(default); 1=pre-load data to memory (make sure memory is enough!)')
	parser.add_argument('--early_stop', type=int, default=50,
						help='Maximum number of unreduced losses allowed (default: 50)')
	parser.add_argument('--local_rank', type=int, default=0,
						help='local rank for DDP (default: 0)')
	parser.add_argument('--debug', type=int, default=0,
						help='To get samples that caused error (0=no, 1=yes; default: 0)')
	parser.add_argument('--pinmemory', type=int, default=1,
						help='pinmemory for dataloader (0=no, 1=yes; default: 1)')
	parser.add_argument('--schedule', type=int, default=25,
						help='steps for schedule of torch.optim.lr_scheduler.StepLR (default: 25)')
	parser.add_argument('--MinLR', type=float, default=0.00001,
						help='min LR for schedule of torch.optim.lr_scheduler.StepLR (default: 0.00001)')

	args = parser.parse_args()
	params = vars(args)
	return params


if __name__ == '__main__':
	params = argparser()
	strTrainFolder = params['F']
	bUseTmpfs = (params['UseTmpfs'] == 1)
	strTrainList = params['L']
	StartEpoch = params['S']
	StartNetFile = params['net']
	LearningRate = params['lr']
	WeightDecay = params['wd']
	TotalEpoch = params['num_epoch']
	batch_size = params['batch_size']
	PlotGap = params['plotgap']
	GPU = params['gpu']
	SHELLs = params['shell']
	NodeEmb = params['emb_dim']
	ffnEmb = params['ffn_emb']
	MLP = params['mlp']
	Heads = params['heads']
	multi_hop_max_dist = params['multi_hop_max_dist']
	dropout = params['dropout']
	num_encoder_layers = params['num_encoder_layers']
	num_producer_threads = params['num_producer_threads']
	LossType = params['loss']
	bUseEdgeFeature = (params['UseEdgeFeature'] == 1)
	bUseDegree = (params['UseDegree'] == 1)
	bCheckData = (params['CheckData'] == 1)
	OPT = params['opt']
	EarlyStop = params['early_stop']
	Preload = (params['preload'] == 1)
	DEBUG = (params['debug'] == 1)
	PINMEMORY = (params['debug'] == 1)
	STEP = params['schedule']
	MinLR = params['MinLR']
	if Preload and num_producer_threads != 0:
		print('reset num_producer_threads=0 for Preload=1!')
		num_producer_threads = 0

	SystemType = sys.platform
	if 'win' in SystemType:
		dist.init_process_group(backend='gloo')
	else:
		dist.init_process_group(backend='nccl')
	local_rank = torch.distributed.get_rank()  # params['local_rank']
	torch.cuda.set_device(local_rank)
	device = torch.device("cuda", local_rank)
	print('local_rank', local_rank)
	NumOfGPUs = torch.cuda.device_count()

	if bUseTmpfs:
		hdfFile = strTrainFolder.split('/')[-1].split('\\')[-1]
		if (not os.path.exists('/dev/shm/GSScore/'+hdfFile)) and ('hdf' in strTrainFolder):
			if not os.path.exists('/dev/shm/GSScore/'):
				os.mkdir('/dev/shm/GSScore')
			shutil.copy(strTrainFolder, '/dev/shm/GSScore/')
			strTrainFolder = '/dev/shm/GSScore/' + hdfFile

	MyNet = MultiShellGraphormer(SHELLs=SHELLs,num_MLP=MLP,num_encoder_layers=num_encoder_layers
								 , multi_hop_max_dist=multi_hop_max_dist
								 , embedding_dim=NodeEmb, ffn_embedding=ffnEmb
								 , num_attention_heads=Heads, dropout=dropout, bUseDegree=bUseDegree)

	Iteration_loss_list = []
	IterationCount = 0
	epoch_mse_list = []
	if (StartEpoch > 0):
		kwargs = {'map_location': lambda storage, loc: storage.cuda('cuda')}
		model = load_MultiGPUS(model=MyNet, model_path=StartNetFile, kwargs=kwargs)
		if False and os.path.exists('IterationLoss.npz'):
			with np.load('IterationLoss.npz') as m:
				Iteration_loss_list = m['arr_0'].tolist()
				IterationCount = len(Iteration_loss_list)
		if False and os.path.exists('EpochLoss.npz'):
			with np.load('EpochLoss.npz') as m:
				epoch_mse_list = m['arr_0'].tolist()
	MyNet = MyNet.cuda()
	MyNet = torch.nn.parallel.DistributedDataParallel(MyNet, find_unused_parameters=True)#, device_ids=[local_rank]
	MyNet.train()
	if OPT == 0:
		optimizer = torch.optim.Adam(MyNet.parameters(), lr=LearningRate, weight_decay=WeightDecay)
	else:
		optimizer = torch.optim.AdamW(MyNet.parameters(), lr=LearningRate, weight_decay=WeightDecay)
	if LossType == 0:
		lossfunct = torch.nn.MSELoss()
		lossLabel = 'MSE loss'
	else:
		lossfunct = torch.nn.L1Loss(reduction="sum")
		lossLabel = 'MAE loss(sum)'

	schedule = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=STEP, gamma=0.5)

	Time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
	print(''.join(['Start at: ', Time]))
	currentLR = schedule.get_last_lr()[0]
	print('current Learning Rate = %f' % (currentLR))
	TrainData = GetGraphData(SHELLs=SHELLs, GraphFile=strTrainFolder, ListFile=strTrainList
							 , max_hop=multi_hop_max_dist, bCheckDataset=bCheckData
							 , bUseEdgeFeature=bUseEdgeFeature, bPreload=Preload, bUseDegree=bUseDegree)
	DataSampler = torch.utils.data.distributed.DistributedSampler(TrainData)
	dataloader = DataLoader(dataset=TrainData
							, batch_size=batch_size
							, num_workers=num_producer_threads
							, shuffle=False
							, pin_memory=PINMEMORY
							, collate_fn=Func_DatasetPostProcessing#)
							, sampler=DataSampler)

	total = TrainData.__len__()
	minEpochLoss = float('inf')
	EarlyStopCount = 0
	if DEBUG:
		batch_file = open('BatchFile.log', 'w')
	for Epoch in range(StartEpoch, (StartEpoch + TotalEpoch)):
		startepoch = time.time()
		if DEBUG:
			print('Epoch %d:' % (Epoch), file=batch_file, flush=True)
		DataSampler.set_epoch(Epoch)
		batch_loss_list = []
		count = 0
		time_list = []
		batch_count = 0
		for [batch_data, sample_ids] in dataloader:
			try:
				startBatch = time.time()
				batch_count += 1
				ExpData = batch_data['ExpData'].cuda()
				count += ExpData.shape[0] * NumOfGPUs
				if DEBUG:
					# batch_file.write('batch %d:' % (batch_count))
					print('batch %d:' % (batch_count),end='',file=batch_file, flush=True)
					for s in sample_ids:
						# batch_file.write('\t%s' % (s))
						print('\t%s' % (s), end='', file=batch_file, flush=True)
					# batch_file.write('\n', flush=True)
					print('\n', end='', file=batch_file, flush=True)
				pred = MyNet(batch_data)
				loss = lossfunct(pred, ExpData)
				# batch_mse = loss.item()*float('nan')
				batch_mse = loss.item()
				if np.isnan(batch_mse): # nan detected
					print('\nbatch %d/%d has Nan error，ignored:' % (count, total), flush=True)
					# torch.save(MyNet.state_dict(), 'Net_Epoch-Nan.pt')
					for s in sample_ids:
						# batch_file.write('\t%s' % (s))
						print('%s' % (s), flush=True)
					# sys.exit(0)
					torch.cuda.empty_cache()
					continue
				Iteration_loss_list.append(batch_mse)
				IterationCount += 1
				batch_loss = batch_mse * pred.shape[0]
				batch_loss_list.append([batch_loss, pred.shape[0]])
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				endBatch = time.time()
				time_list.append(endBatch - startBatch)
				TimeCost = np.mean(time_list)
				print('\rFinished batch %d/%d, %s=%f, time remaining %.2f seconds ...'
					  % (count, total, lossLabel, batch_mse, TimeCost * (total - count) / (batch_size * NumOfGPUs)), end='', flush=True)

				if IterationCount % PlotGap == 0 and IterationCount > 100:
					x_values = np.arange(start=100, stop=IterationCount + 1, step=PlotGap)
					y_values = np.array(Iteration_loss_list[100 - 1::PlotGap])
					plt.clf()
					plt.xlabel('Iterations')
					plt.ylabel(lossLabel)
					plt.title('Training Iteration')
					plt.plot(x_values, y_values, 'r-', label='Train loss')
					plt.legend()
					# plt.show()
					# plt.pause(0.01)
					plt.savefig('IterationLoss.png')
			except Exception as err:
				print('Error samples:')
				for s in sample_ids:
					# batch_file.write('\t%s' % (s))
					print('%s' % (s), flush=True)
				# sys.exit(0)
				torch.cuda.empty_cache()

		currentLR = schedule.get_last_lr()[0]
		if currentLR >= MinLR: #LR schedule
			schedule.step()
		endepoch = time.time()
		epoch_loss = np.array(batch_loss_list)
		temp_epoch_loss = np.sum(epoch_loss, axis=0)
		if temp_epoch_loss[0] < minEpochLoss:
			minEpochLoss = temp_epoch_loss[0]
			EarlyStopCount = 0
		else:
			EarlyStopCount += 1
		epoch_mse_list.append(temp_epoch_loss[0] / temp_epoch_loss[1])
		x_values = np.arange(start=StartEpoch + 1, stop=Epoch + 1 + 1)
		y_values = np.array(epoch_mse_list)
		plt.clf()
		plt.xlabel('Epoch')
		plt.ylabel(lossLabel)
		plt.title('Training epoch')
		plt.plot(x_values, y_values, 'ro-', label='Train loss')
		plt.legend()
		plt.savefig('EpochLoss.png')
		# save Iteration and Epoch Loss
		np.savez_compressed('IterationLoss.npz', np.array(Iteration_loss_list))
		np.savez_compressed('EpochLoss.npz', np.array(epoch_mse_list))
		# Save network checkpoint
		NetSaveName = ''.join(['Net_Epoch-', str(Epoch + 1), '.pt'])
		# CheckPoint = {'model': MyNet.state_dict()}
		# torch.save(CheckPoint, NetSaveName)
		torch.save(MyNet.state_dict(), NetSaveName)
		endEpoch = time.time()
		print('\rFinished batch %d/%d, %s=%f, time remaining %.2f seconds ...'
			  % (count, total, lossLabel, y_values[-1], 0.0), end='', flush=True)
		print('Finished Epoch %d within %.2f seconds.' % (Epoch, endepoch - startepoch), flush=True)
		if EarlyStopCount > EarlyStop:
			print('Early stop detected for Max number = %d!' % (EarlyStop), flush=True)
			break

	if DEBUG:
		batch_file.close()
	# remove files in tmpfs
	try:
		shutil.rmtree('/dev/shm/GSScore')
	except Exception as err:
		print('removing tmpfs error: %s' % err)
	print('Finished ALL!')
