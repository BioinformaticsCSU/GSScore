# To predict the RMSD of protein-ligand conformation

import sys, os, argparse
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import time

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
		self.PDBNames = np.loadtxt(ListFile, skiprows=0, usecols=[0], dtype=str)  # 以str类型获取DataFileName第一列
		self.PoseNames = np.loadtxt(ListFile, skiprows=0, usecols=[1], dtype=str)  # 以str类型获取DataFileName第二列
		self.ExpDatas = np.loadtxt(ListFile, skiprows=0, usecols=[2], dtype=float)  # 以float类型获取DataFileName第三列
		self.bUseEdgeFeature = bUseEdgeFeature
		self.PreloadData = {}
		self.bPreload = bPreload
		self.bUseDegree = bUseDegree
		# self.HDF5 = h5py.File(self.GraphFile, 'r')
		if bCheckDataset:
			self.Func_CheckDataset()

	def Func_CheckDataset(self):
		print('Checking data integrity ...')
		self.HDF5 = h5py.File(self.GraphFile, 'r')
		newPDBNames = []
		newPoseNames = []
		newExpDatas = []
		total = len(self.PoseNames)
		for index in range(len(self.PoseNames)):
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
				print('\r%d/%d ...' % (index + 1, total), end='', flush=True)
		self.PDBNames = np.array(newPDBNames)
		self.PoseNames = np.array(newPoseNames)
		self.ExpDatas = np.array(newExpDatas)
		self.HDF5.close()
		self.HDF5 = None
		f = open('TestCurrentDecoys.list', 'w')
		for i in range(self.PDBNames.shape[0]):
			f.write('%s %-8s %f\n' % (self.PDBNames[i], self.PoseNames[i], self.ExpDatas[i]))
		f.close()
		print('Finished checking!')

	def Func_GetPDBList(self):
		return self.PDBNames, self.PoseNames, self.ExpDatas

	def __getitem__(self, index):
		PDBName = self.PDBNames[index]
		PoseName = self.PoseNames[index]
		ExpData = self.ExpDatas[index].reshape(-1).astype(np.float32)
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
			tempLigandAtomFeatures = torch.zeros([num_atoms, sum(offset_list) + 2], dtype=torch.int8)
			sum_k = 0
			for index, k in enumerate(offset_list):
				sum_k += k
				tempLigandAtomFeatures.scatter_(-1, LigandAtomFeatures[:, index:index + 1] + sum_k, 1)
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
			for shell in range(1, self.SHELLs + 1):
				TempGroup = Group.require_group(MyDebug + '/' + str(shell))
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
				tempProteinAtomFeatures = torch.zeros([num_atoms, sum(offset_list) + 37], dtype=torch.int8)
				sum_k = 0
				for i, k in enumerate(offset_list):
					sum_k += k
					tempProteinAtomFeatures.scatter_(-1, ProteinAtomFeatures[:, i:i + 1] + sum_k, 1)
				# check = (tempProteinAtomFeatures==tempProteinAtomFeatures1).all()
				ProteinAtomFeatures = tempProteinAtomFeatures

				AdjacencyDistMatrix = torch.Tensor(TempGroup["AdjacencyDistMatrix"][:])
				AdjacencyMatrix = torch.Tensor(TempGroup["AdjacencyMatrix"][:])
				if (AdjacencyMatrix is not None) and self.bUseDegree:
					t = (AdjacencyMatrix > 0.0) * (AdjacencyMatrix <= 6.0) * 1
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

			return attn_bias_list, spatial_pos_list, x_list, edge_input_list, degree_list \
				, [AtomFeatureSize, hop_max, edge_size], NumAtomOfEachShell \
				, PDBName + ' ' + PoseName + ' ' + str(ExpData), ExpData
		except Exception as err:
			print('ERROR in %s' % (MyDebug))
			print(err)
			traceback.print_exc()
			sys.exit()

	def __len__(self):
		return (self.ExpDatas.shape[0])


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
			return_data[shell + 1] = {}
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
					attn_bias[index][:, :tempY] = 0  # decoy[0][shell]
					check = True
					if (attn_bias[index] == float('-inf')).all():
						check = False
				if decoy[1][shell] is not None:
					tempX, tempY = decoy[1][shell].shape[:2]
					spatial_pos[index][:tempX, :tempY] = decoy[1][shell]
					check = True
					if (spatial_pos[index] == float('inf')).any():
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

			return_data[shell + 1]['attn_bias'] = attn_bias[listOfAvailable]
			return_data[shell + 1]['spatial_pos'] = spatial_pos[listOfAvailable]
			return_data[shell + 1]['x'] = x[listOfAvailable]
			return_data[shell + 1]['edge_input'] = edge_input[listOfAvailable]
			return_data[shell + 1]['in_degree'] = in_degree[listOfAvailable]
			return_data[shell + 1]['out_degree'] = out_degree[listOfAvailable]
			return_data[shell + 1]['hasGraph'] = hasGraph

	return return_data, sample_ids


def load_MultiGPUS(model, map_location, model_path, kwargs):
	# state_dict = torch.load(model_path, **kwargs)
	# # create new OrderedDict that does not contain `module.`
	# from collections import OrderedDict
	# new_state_dict = OrderedDict()
	# for k, v in state_dict['module'].items():
	# 	name = k[7:] # remove `module.`
	# 	new_state_dict[name] = v
	# 	# load params
	# model.load_state_dict(new_state_dict)
	# return model.cuda()
	state_dict = torch.load(model_path, map_location=map_location)
	model = nn.DataParallel(model)
	model = model.cuda()
	model.load_state_dict(state_dict, strict=False)
	return model


# Define function to calculate the performance of network
def CalNetPerformance(strNetFile
					  , strTestFolder
					  , strListFile
					  , SHELLs
					  , NodeEmb
					  , ffn_embedding
					  , Heads
					  , MLP
					  , multi_hop_max_dist
					  , dropout
					  , num_encoder_layers
					  , BenchSize
					  , BenchNumWorkers
					  , OutFileName
					  , bUseEdgeFeature, bUseDegree, GPU):
	if not os.path.exists(OutFileName):
		model = MultiShellGraphormer(SHELLs=SHELLs, num_MLP=MLP, num_encoder_layers=num_encoder_layers
									 , multi_hop_max_dist=multi_hop_max_dist
									 , embedding_dim=NodeEmb, ffn_embedding=ffn_embedding
									 , num_attention_heads=Heads, dropout=dropout, bUseDegree=bUseDegree)
		mp = {'cuda:7': 'cuda:0'
			, 'cuda:6': 'cuda:0'
			, 'cuda:5': 'cuda:0'
			, 'cuda:4': 'cuda:0'
			, 'cuda:3': 'cuda:0'
			, 'cuda:2': 'cuda:0'
			, 'cuda:1': 'cuda:0'
			, 'cuda:0': 'cuda:0'}
		try:
			Temp = torch.load(strNetFile
							  , map_location=mp)
			model.load_state_dict(Temp)
		except Exception as err:
			# print(err)
			print('Try to load MultiGPUs!')
			kwargs = {'map_location': lambda storage, loc: storage.cuda('cuda')}
			model = load_MultiGPUS(model=model, map_location=mp, model_path=strNetFile, kwargs=kwargs)
		model = model.eval()
		if len(GPU.split(',')) > 1:
			choose = [_ for _ in range(len(GPU.split(',')))]
			model = nn.DataParallel(model, device_ids=choose)
		Net = model.cuda()
		with torch.no_grad():
			TestData = GetGraphData(SHELLs=SHELLs, GraphFile=strTestFolder, ListFile=strListFile
									, max_hop=multi_hop_max_dist, bUseEdgeFeature=bUseEdgeFeature)
			# DataSampler = torch.utils.data.distributed.DistributedSampler(TrainData)
			dataloader = DataLoader(dataset=TestData
									, batch_size=BenchSize
									, num_workers=BenchNumWorkers
									, shuffle=False
									, drop_last=False
									, collate_fn=Func_DatasetPostProcessing)
			count = 0
			TempExp = TestData.Func_GetPDBList()[-1]
			AvgTempCal = np.zeros_like(TempExp)
			TotalPose = TempExp.shape[0]
			index = 0
			time_list = []
			for [batch_data, sample_ids] in dataloader:
				startBatch = time.time()
				ExpData = batch_data['ExpData']
				DataNum = ExpData.shape[0]
				pred = Net(batch_data)
				debug = pred.cpu().numpy()
				if np.isnan(debug).any():  # nan samples detected
					print('\nbatch %d/%d has Nan result, ignored:' % (count, TotalPose), flush=True)
					# torch.save(MyNet.state_dict(), 'Net_Epoch-Nan.pt')
					for s in sample_ids:
						# batch_file.write('\t%s' % (s))
						print('%s' % (s), flush=True)
					# sys.exit(0)
					torch.cuda.empty_cache()
					continue
				AvgTempCal[index: (index + DataNum)] = pred.cpu().detach().numpy().transpose()
				count += DataNum
				index += DataNum
				endBatch = time.time()
				time_list.append(endBatch - startBatch)
				TimeCost = np.mean(time_list)
				print('\rFinished %d/%d time remaining %.2f seconds ...'
					  % (count, TotalPose, TimeCost * (TotalPose - count) / BenchSize), end='', flush=True)
		# Print out experimental and calculated values
		FileID = open(OutFileName, "w")
		for n in range(TotalPose):
			print('%8.4f|%8.4f' % (TempExp[n], AvgTempCal[n]), file=FileID)

		FileID.close()
	else:
		print('results was already existed')


def argparser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-N', type=str, required=True, help='network parameters from checkpoint')
	parser.add_argument('-T', type=str, required=True, help='HDF5 input file merged from *.h5 (*.hdf5)')
	parser.add_argument('-L', type=str, required=True, help='List of pose (Pose.dat)')
	parser.add_argument('--shell', type=int, default=10
						,
						help='number of shells to be used in each decoy, (default: 10)')
	parser.add_argument('--heads', type=int, default=8, help='number of heads in attention (default: 8)')
	parser.add_argument('--emb_dim', type=int, default=128
						,
						help='dimension of atom embedding vector, (default: 128)')
	parser.add_argument('--ffn_emb', type=int, default=128
						,
						help='dimension of ffn embedding, (default: 128)')
	parser.add_argument('--mlp', type=int, default=1
						,
						help='number of final mlp layers, (default: 1)')
	parser.add_argument('--multi_hop_max_dist', type=int, default=8, help='number of hop_max_dist (default: 8)')
	parser.add_argument('--dropout', type=float, default=0.0, help='dropout in Fairseq (default: 0.0)')
	parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
	parser.add_argument('--num_producer_threads', type=int, default=4
						, help='number of producer threads (default: 4)')
	parser.add_argument('--num_encoder_layers', type=int, default=16, help='num of encoder (default: 16)')
	parser.add_argument('--UseEdgeFeature', type=int, default=1,
						help='0=do NOT use edge features in Graphormer; 1=Yes!Use it!(default)')
	parser.add_argument('--UseDegree', type=int, default=0,
						help='0=do NOT use degree in Graphormer(default); 1=Yes!Use it!')
	parser.add_argument('--gpu', type=str, default='0',
						help='Choose gpu id, example: \'0,1\'(specify use gpu 0, 1 or any other)')
	parser.add_argument('--local_rank', type=int, default=0,
						help='local rank for DDP (default: 0)')
	args = parser.parse_args()
	params = vars(args)
	return params


if __name__ == '__main__':
	params = argparser()
	strNetFile = params['N']
	strTestFolder = params['T']
	strListFile = params['L']
	SHELLs = params['shell']
	NodeEmb = params['emb_dim']
	ffnEmb = params['ffn_emb']
	Heads = params['heads']
	MLP = params['mlp']
	multi_hop_max_dist = params['multi_hop_max_dist']
	dropout = params['dropout']  # default: 0.0
	num_encoder_layers = params['num_encoder_layers']
	batch_size = params['batch_size']
	NumOfProducer = params['num_producer_threads']
	UseEdgeFeature = (params['UseEdgeFeature'] == 1)
	bUseDegree = (params['UseDegree'] == 1)
	GPU = params['gpu']
	os.environ["CUDA_VISIBLE_DEVICES"] = GPU

	OutFileName = os.path.basename(strNetFile) + '.out'
	print('output file: ', os.path.abspath(OutFileName))

	CalNetPerformance(strNetFile=strNetFile
					  , strTestFolder=strTestFolder
					  , strListFile=strListFile
					  , SHELLs=SHELLs
					  , NodeEmb=NodeEmb
					  , ffn_embedding=ffnEmb
					  , Heads=Heads
					  , MLP=MLP
					  , multi_hop_max_dist=multi_hop_max_dist
					  , dropout=dropout
					  , num_encoder_layers=num_encoder_layers
					  , BenchSize=batch_size
					  , BenchNumWorkers=NumOfProducer
					  , OutFileName=OutFileName
					  , bUseEdgeFeature=UseEdgeFeature
					  , bUseDegree=bUseDegree
					  , GPU=GPU)

	print('Finished ALL!')
