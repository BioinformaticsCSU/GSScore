#To prepare training or testing data

import sys, os, argparse
import numpy as np
import rdkit
from rdkit import Chem
import openbabel
import AtomType
import multiprocessing, traceback
from scipy.sparse.csgraph import floyd_warshall
import h5py
from rdkit.Chem import rdPartialCharges

def argparser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-F', type=str, required=True, help='the pdb folder, Make sure that there are ProteinList.txt')
	parser.add_argument('-O', type=str, default='', help='the output folder (default: [Input]_out)')
	parser.add_argument('-P', type=int, default=1, help='the number of processes (default: 1)')
	parser.add_argument('-S', type=int, default=30, help='the number of shells (default: 30)')
	parser.add_argument('--list', type=str, default=''
						, help='the list file for decoys to process (default: ProteinList.txt)')
	parser.add_argument('--max_hop', type=int, default=10
						, help='the max number of hopping for shortest path record (default: 10)')
	# parser.add_argument('--rec_dist', type=float, default=6.0
	# 					, help='the max distance of receptor atoms from ligand (default: 6.0)')
	parser.add_argument('--tmpfs', type=str, default='/dev/shm/'
						, help='the tmpfs of Linux or RAMDisk of Windows for rdkit, (default: /dev/shm/)')
	parser.add_argument('--gap', type=float, default=0.5
						, help='the gap of each shell (default: 0.5)')
	parser.add_argument('--d0', type=float, default=5.0
						, help='the base distance for the first shell (default: 5.0)')
	args = parser.parse_args()
	params = vars(args)
	return params


# list of Lig_*.pdb
def Bank_file_name(file_dir):
	L = []
	for root, dirs, files in os.walk(file_dir):
		for file in files:
			tempFile = os.path.splitext(file)
			if tempFile[1].lower() == '.pdb':
				L.append(file)
	return L


#read Rec.pdb and Lig_*.pdb
def Func_ReadPDB(proteinList, strProteinFolder, tempPath='/dev/shm/'):
	if type(proteinList) == str:
		protein = proteinList
		receptor = Chem.MolFromPDBFile(strProteinFolder + '/' + protein + '/Rec.pdb', sanitize=False)
		Lig_list = Bank_file_name(strProteinFolder + '/' + protein)
	elif type(proteinList) == list:
		protein = proteinList[0]
		Lig_list = proteinList[1]
		receptor = Chem.MolFromPDBFile(strProteinFolder + '/' + protein + '/Rec.pdb', sanitize=False)
	else:
		print('ERROR ProteinList!')
		sys.exit()

	lig_native = None
	listOfDecoys = []
	for lig in Lig_list:
		if 'Lig' == lig[:3]:
			try:
				conv = openbabel.OBConversion()
				ob = openbabel.OBMol()
				conv.SetInAndOutFormats('pdb', 'pdb')
				conv.ReadFile(ob, strProteinFolder + '/' + protein + '/' + lig)
				ob.DeleteHydrogens()  # 是不是不应该删除氢原子？
				conv.Convert()
				conv.WriteFile(ob, tempPath + '/' + protein + '-' + lig)
				del conv
				del ob
				conv = openbabel.OBConversion()
				ob = openbabel.OBMol()
				f = open(tempPath + '/' + protein + '-' + lig)
				lines = f.readlines()
				f.close()
				f = open(tempPath + '/' + protein + '-' + lig, 'w')
				for l in lines:
					if l[:4] == 'ATOM' or l[:6] == 'HETATM':
						f.write('HETATM%s' % (l[6:]))
				f.close()
				conv.SetInAndOutFormats('pdb', 'mol')
				conv.ReadFile(ob, tempPath + '/' + protein + '-' + lig)
				conv.Convert()
				conv.WriteFile(ob, tempPath + '/' + protein + '-' + lig + '.mol')
				del conv
				del ob
				decoy = Chem.MolFromMolFile(tempPath + '/' + protein + '-' + lig + '.mol', removeHs=True
											, strictParsing=False, sanitize=False)
				r = Chem.SanitizeMol(decoy
									 , sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
												   ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
				if decoy is None:
					print('ERROR in reading %s' % (strProteinFolder + '/' + protein + '/' + lig))
				else:
					# To remove isolative atom
					decoy=rdkit.Chem.rdchem.RWMol(decoy)
					AdjacencyMatrix = Chem.rdmolops.GetAdjacencyMatrix(decoy)
					AdjacencyMatrix = np.array(AdjacencyMatrix)
					t1 = AdjacencyMatrix.sum(axis=0)
					# t2 = AdjacencyMatrix.sum(axis=1)
					indexList1 = np.array(np.where(t1 == 0))[0].tolist()
					indexList1.sort(reverse=True)
					for index in indexList1:
						rdkit.Chem.rdchem.RWMol.RemoveAtom(decoy, index)
					# Chem.SanitizeMol(decoy)
					# AdjacencyMatrix2 = np.array(Chem.rdmolops.GetAdjacencyMatrix(decoy))
					# check=(AdjacencyMatrix2==AdjacencyMatrix).all()
					###############################################################################
					listOfDecoys.append([lig, decoy])
					##DEBUG
					# print('%s Bonds:' % (tempPath + '/' + protein + '-' + lig))
					# bonds = decoy.GetBonds()
					# for b in bonds:
					# 	name = b.GetBondType().name
					# 	print(name)
				if ('native' in lig) or ('Lig_0' == lig[:5]):
					lig_native = decoy
				os.remove(tempPath + '/' + protein + '-' + lig)
				os.remove(tempPath + '/' + protein + '-' + lig + '.mol')
			except Exception as err:
				print('ERROR in reading %s' % (strProteinFolder + '/' + protein + '/' + lig))
				print(err)
				traceback.print_exc()
				# del conv
				os.remove(tempPath + '/' + protein + '-' + lig)
				# os.remove(tempPath + '/' + protein + '-' + lig + '.mol')

	return receptor, lig_native, listOfDecoys


def Func_DumpHDF(HDF5File:h5py.File, data:dict):
	for lig in data.keys():
		LIG = HDF5File.create_group(''.join(['/', lig, '/']))
		for shell in data[lig].keys():
			if shell=='LigandAtomFeatures':
				d = data[lig]['LigandAtomFeatures']
				LigandAtomFeatures = []
				keys = [k for k in d.keys()]
				keys.sort()
				for k in keys:
					LigandAtomFeatures.append(d[k])
				LigandAtomFeatures = np.array(LigandAtomFeatures)
				LIG.create_dataset(name='LigandAtomFeatures'
								   #, dtype='i2'
								   , data=LigandAtomFeatures
								   , compression="gzip"
								   , compression_opts=9)
				continue
			d = data[lig][shell]
			LIG_shell = LIG.create_group(''.join([str(shell), '/']))
			ProteinAtomFeatures = []
			keys = [k for k in d['ProteinAtomFeatures'].keys()]
			keys.sort()
			for k in keys:
				ProteinAtomFeatures.append(d['ProteinAtomFeatures'][k])
			ProteinAtomFeatures = np.array(ProteinAtomFeatures)
			LIG_shell.create_dataset(name='ProteinAtomFeatures'
							   , dtype='i2'
							   , data=ProteinAtomFeatures
							   , compression="gzip"
							   , compression_opts=9)

			LIG_shell.create_dataset(name='AdjacencyMatrix'
							   , data=d['AdjacencyMatrix']
							   , compression="gzip"
							   , compression_opts=9)

			LIG_shell.create_dataset(name='AdjacencyDistMatrix'
							   , data=d['AdjacencyDistMatrix']
							   , compression="gzip"
							   , compression_opts=9)

			LIG_shell.create_dataset(name='edge_input'
							   , dtype='i2'
							   , data=d['edge_input']
							   , compression="gzip"
							   , compression_opts=9)


# To get protein atom features
def Func_GetProteinFeatures(receptor: rdkit.Chem.rdchem.Mol#, ligand: rdkit.Chem.rdchem.Mol
							, protein:str, strProteinFolder:str):
	if receptor is None:
		return None
	f = open(strProteinFolder + '/' + protein + '/Rec.pdb')
	lines = f.readlines()
	f.close()
	coord_map = {}
	for l in lines:
		if l[:4] == 'ATOM' or l[:6] == 'HETATM':
			AtomName = l[12:16].strip()
			x = float(l[30:38].strip())
			y = float(l[38:46].strip())
			z = float(l[46:54].strip())
			coord_key = str(x) + '_' + str(y) + '_' + str(z)
			coord_map[coord_key] = AtomName
	conformer = receptor.GetConformer()
	listOfAtom = receptor.GetAtoms()
	# AtomsIdxInBox = []
	AtomsIdxInBox = {}
	AllProteinAtomFeatures = {}
	for atom in listOfAtom:
		position = conformer.GetAtomPosition(atom.GetIdx())
		coord_key = str(position.x) + '_' + str(position.y) + '_' + str(position.z)
		AtomName = coord_map[coord_key]
		AtomsIdxInBox[atom.GetIdx()] = np.array([position.x, position.y, position.z])
		if AtomName in AtomType.ProteinAtomName_num:
			AtomName_vec = AtomType.ProteinAtomName_num[AtomName]
		else:
			AtomName_vec = AtomType.ProteinAtomName_num['unk']
		symbol = atom.GetSymbol().upper()
		if not (symbol in AtomType.ProteinAtomType_num):
			symbol = 'unk'
		symbol_vec = AtomType.ProteinAtomType_num[symbol]
		degree = atom.GetDegree()
		if not (degree in AtomType.ProteinAtomDegree_num):
			degree = 'unk'
		degree_vec = AtomType.ProteinAtomDegree_num[degree]
		ImplicitValence = atom.GetImplicitValence()
		if not (ImplicitValence in AtomType.ProteinImpliciteValence_num):
			ImplicitValence = 'unk'
		ImplicitValence_vec = AtomType.ProteinImpliciteValence_num[ImplicitValence]
		NeighboringHs = atom.GetTotalNumHs()
		if not (NeighboringHs in AtomType.ProteinNeighboringHs_num):
			NeighboringHs = 'unk'
		NeighboringHs_vec = AtomType.ProteinNeighboringHs_num[NeighboringHs]
		Hybridization = atom.GetHybridization().name
		if not (Hybridization in AtomType.ProteinAtomHybridization_num):
			Hybridization = 'unk'
		Hybridization_vec = AtomType.ProteinAtomHybridization_num[Hybridization]
		Residue = atom.GetPDBResidueInfo()
		if Residue is not None:
			res_name = Residue.GetResidueName()
			if not (res_name in AtomType.AminoAcidtoAbbreviation):
				res = 'unk'
			else:
				res = AtomType.AminoAcidtoAbbreviation[res_name]
			Residue_vec = AtomType.ProteinAminoAcidType_num[res]
		else:
			Residue_vec = [0] * 22
		ProteinAtomFeatures = [symbol_vec, degree_vec, ImplicitValence_vec, NeighboringHs_vec
			, Hybridization_vec, Residue_vec, AtomName_vec]
		AllProteinAtomFeatures[atom.GetIdx()] = np.array(ProteinAtomFeatures, dtype=np.int8)
	# AtomsIdxInBox.sort(key=lambda d: d[0])
	return AllProteinAtomFeatures, AtomsIdxInBox


# To get ligand atom features
def Func_GetLigandFeatures(mol: rdkit.Chem.rdchem.Mol, protein:str, lig:str):
	if mol is None:
		return None
	rdPartialCharges.ComputeGasteigerCharges(mol)
	listOfAtom = mol.GetAtoms()
	RingInfo = mol.GetRingInfo()
	AtomRings = RingInfo.AtomRings()
	AllLigandFeatures = {}
	for atom in listOfAtom:
		symbol = atom.GetSymbol()
		if not (symbol in AtomType.LigandAtomType_num):
			symbol = 'unk'
		symbol_vec = AtomType.LigandAtomType_num[symbol]
		degree = atom.GetDegree()
		if not (degree in AtomType.LigandAtomDegree_num):
			degree = 'unk'
		degree_vec = AtomType.LigandAtomDegree_num[degree]
		ImplicitValence = atom.GetImplicitValence()
		if not (ImplicitValence in AtomType.LigandImpliciteValence_num):
			ImplicitValence = 'unk'
		ImplicitValence_vec = AtomType.LigandImpliciteValence_num[ImplicitValence]
		NeighboringHs = atom.GetTotalNumHs()
		if not (NeighboringHs in AtomType.LigandNeighboringHs_num):
			NeighboringHs = 'unk'
		NeighboringHs_vec = AtomType.LigandNeighboringHs_num[NeighboringHs]
		Hybridization = atom.GetHybridization().name
		if not (Hybridization in AtomType.LigandAtomHybridization_num):
			Hybridization = 'unk'
		Hybridization_vec = AtomType.LigandAtomHybridization_num[Hybridization]
		FormalCharge = atom.GetFormalCharge()
		if not (FormalCharge in AtomType.LigandFormalCharge_num):
			FormalCharge = 'unk'
		FormalCharge_vec = AtomType.LigandFormalCharge_num[FormalCharge]
		ringOfsize = 0
		for ring in AtomRings:
			if atom.GetIdx() in ring:
				if len(ring) > ringOfsize:
					ringOfsize = len(ring)
		if ringOfsize in AtomType.LigandRingSize_num:
			RingSize_vec = AtomType.LigandRingSize_num[ringOfsize]
		else:
			RingSize_vec = AtomType.LigandRingSize_num['unk']
		AromaticType_vec = AtomType.LigandAromaticType_num[atom.GetIsAromatic()]
		GasteigerCharge = atom.GetProp("_GasteigerCharge")
		LigandAtomFeatures = [symbol_vec, degree_vec, ImplicitValence_vec, NeighboringHs_vec
			, Hybridization_vec, FormalCharge_vec, RingSize_vec, AromaticType_vec, float(GasteigerCharge)]
		if GasteigerCharge.upper() == 'NAN':
			conformer = mol.GetConformer()
			position = conformer.GetAtomPosition(atom.GetIdx())
			print('发现nan: %s/%s-%s%s->GasteigerCharge=%s'
				  % (protein, lig, symbol, [position.x, position.y, position.z], GasteigerCharge))
		# LigandAtomFeatures = [symbol_vec, degree_vec, ImplicitValence_vec, NeighboringHs_vec
		# 	, Hybridization_vec, FormalCharge_vec, RingSize_vec, AromaticType_vec]
		AllLigandFeatures[atom.GetIdx()] = np.array(LigandAtomFeatures, dtype=np.float32)
	return AllLigandFeatures


#To get edge features
def Func_GetEdgeFeatures(ProteinAtomsInBox: dict, ligand: rdkit.Chem.rdchem.Mol, ConnectedMatrix:np.ndarray
						 , DistMin:float, DistMax:float):
	# Use rdkit.Chem.rdmolops.GetAdjacencyMatrix to construct adjacency matrix
	AdjacencyMatrix = ConnectedMatrix #Chem.rdmolops.GetAdjacencyMatrix(ligand)
	LigandAtoms = ligand.GetAtoms()
	LigandAtomNum = len(LigandAtoms)
	conformer = ligand.GetConformer()
	listOfLigandAtomPositions = []
	for index, atom in enumerate(LigandAtoms):
		position = conformer.GetAtomPosition(atom.GetIdx())
		listOfLigandAtomPositions.append([atom.GetIdx(), np.array([position.x, position.y, position.z])])
	listOfLigandAtomPositions.sort(key=lambda d: d[0])
	EdgeType = {}
	ProteinAtomsInShell_list = []
	for p_atomIdx in ProteinAtomsInBox: #ProteinAtomsInBox是dict，(atomIdx):[x,y,z]
		p_pos = ProteinAtomsInBox[p_atomIdx]
		check = False
		for l_index, l_atom in enumerate(listOfLigandAtomPositions): #listOfLigandAtomPositions: list，[(atomIdx),[x,y,z]]
			l_pos = l_atom[1]
			l_atomIdx = l_atom[0]
			dist = np.linalg.norm(p_pos - l_pos, ord=2)
			if DistMin <= dist and dist <= DistMax: #在特定的shell范围内
				check = True
				EdgeType['P' + str(p_atomIdx) + '-L' + str(l_atomIdx)] = AtomType.CovalentBondType_num['NonCovalent']
		if check:
			# ProteinAtomsInShell_num += 1
			ProteinAtomsInShell_list.append(p_atomIdx)

	ProteinAtomsInShell_list.sort()
	ProteinAtomsInShell_num = len(ProteinAtomsInShell_list)
	mapProteinAtomIdx2Index = {} # map idx of protein atom to index of ComplexAdjMatrix
	for index, p_atomIdx in enumerate(ProteinAtomsInShell_list):
		ProteinAtomsInBox.pop(p_atomIdx)
		mapProteinAtomIdx2Index[p_atomIdx] = index
	mapLigandAtomIdx2Index = {}  # map idx of ligand atom to index of ComplexAdjMatrix
	for index, l_atom in enumerate(listOfLigandAtomPositions):
		l_atomIdx = l_atom[0]
		mapLigandAtomIdx2Index[l_atomIdx] = ProteinAtomsInShell_num + index
	ComplexAdjMatrix = \
		np.zeros([ProteinAtomsInShell_num + LigandAtomNum, ProteinAtomsInShell_num + LigandAtomNum], dtype=np.float32)
	ComplexAdjMatrix[ProteinAtomsInShell_num:, ProteinAtomsInShell_num:] = AdjacencyMatrix
	for edge in EdgeType:
		tempE = edge.split('-')
		p_atomIdx = int(tempE[0][1:])
		p_index = mapProteinAtomIdx2Index[p_atomIdx]
		l_atomIdx = int(tempE[1][1:])
		l_index = mapLigandAtomIdx2Index[l_atomIdx]
		ComplexAdjMatrix[p_index, l_index] = DistMax #1
		ComplexAdjMatrix[l_index, p_index] = DistMax #1

	# covalent bond in ligand
	bonds = ligand.GetBonds()
	for bond in bonds:
		a = bond.GetBeginAtom().GetIdx()
		b = bond.GetEndAtom().GetIdx()
		name = bond.GetBondType().name
		if not (name in AtomType.CovalentBondType_num):
			name = 'unk'
		EdgeType['L' + str(a) + '-L' + str(b)] = AtomType.CovalentBondType_num[name]
	return ComplexAdjMatrix, EdgeType, mapProteinAtomIdx2Index, mapLigandAtomIdx2Index


# Use Floyd to get the shortest paths between any nodes
#AdjacencyDistMatrix: the shortest distance between any node
#PathMatrix: the shortest path between any node
def Func_Floyd(DistMatrix:np.ndarray):
	AdjacencyMatrix = DistMatrix# DistMatrix.copy()
	AdjacencyDistMatrix, PathMatrix = floyd_warshall(AdjacencyMatrix, directed=False, return_predecessors=True)

	return AdjacencyDistMatrix.astype(np.float32), PathMatrix



def Callback_WriteERRORProtein(data):
	global global_index, global_total
	check = data[0]
	protein = data[1]
	filename = data[2]
	global_index += 1
	print('\rFinished %d/%d ...' % (global_index, global_total), end='')
	if check == False:
		with open(filename, 'a+') as f:
			f.write('%s\n' % (protein))



from sklearn.neighbors import KDTree
def Func_ConstructConnectedMatrix(listOfMol: list, mol: rdkit.Chem.rdchem.Mol):
	ConnectedMatrix = np.asarray(Chem.rdmolops.GetAdjacencyMatrix(mol=mol), dtype=np.float32)

	for i in range(len(listOfMol)):
		atomList1 = list(listOfMol[i])
		atomList2 = []
		# Since there may be multiple subgraphs, after taking all nodes of one subgraph
		# , all nodes of the remaining subgraphs are planned into another subset
		# to avoid repeating the calculation of the connectivity of the two subgraphs
		for j in range(0,i):
			atomList2+=list(listOfMol[j])
		for j in range(i+1,len(listOfMol)):
			atomList2+=list(listOfMol[j])
		conformer = mol.GetConformer()
		L_coords = []
		for atomIndex in atomList1:
			position = conformer.GetAtomPosition(atomIndex)
			L_coords.append([position.x, position.y, position.z, atomIndex])
		L_coords = np.array(L_coords)
		R_coords = []
		for atomIndex in atomList2:
			position = conformer.GetAtomPosition(atomIndex)
			R_coords.append([position.x, position.y, position.z, atomIndex])
		R_coords = np.array(R_coords)
		tree1 = KDTree(L_coords[:, :3], leaf_size=2)
		tree2 = KDTree(R_coords[:, :3], leaf_size=2)
		dist1, indices1 = tree1.query(R_coords[:, :3], k=1, return_distance=True, sort_results=True)
		dist2, indices2 = tree2.query(L_coords[:, :3], k=1, return_distance=True, sort_results=True)

		A = int(L_coords[indices1[np.argmin(dist1)], -1])
		dist_LtoR = dist1[np.argmin(dist1)]
		B = int(R_coords[indices2[np.argmin(dist2)], -1])
		dist_RtoL = dist2[np.argmin(dist2)]
		if dist_LtoR == dist_RtoL:
			dist_LtoR = np.round(dist_LtoR)
			dist_RtoL = np.round(dist_RtoL)
			if ConnectedMatrix[A, B]==0 or ConnectedMatrix[B, A]==0:
				if dist_LtoR >= 1.0:
					ConnectedMatrix[A, B] = dist_LtoR
					ConnectedMatrix[B, A] = dist_RtoL
				else:
					ConnectedMatrix[A, B] = 1.0
					ConnectedMatrix[B, A] = 1.0
			else:
				if dist_LtoR >= 1.0:
					ConnectedMatrix[A, B] = min(dist_LtoR, ConnectedMatrix[A, B])
					ConnectedMatrix[B, A] = min(dist_LtoR, ConnectedMatrix[B, A])
				else:
					ConnectedMatrix[A, B] = min(1.0, ConnectedMatrix[A, B])
					ConnectedMatrix[B, A] = min(1.0, ConnectedMatrix[B, A])
		else:
			print('It is found that the Euclidean distance '
				  'between the shortest nodes in the two subgraphs L and R is inconsistent'
				  ', which corresponds to the atom numberL:%d\tR:%d' % (A, B))

	return ConnectedMatrix


# To start multiple processing
def Func_MultiProcess(proteinList, strProteinFolder: str, strOutputFolder: str
					  , max_hop:int, strTmpfs: str, ShellNum:int, GAP:float, D0:float):
	try:
		if type(proteinList) == list:
			protein = proteinList[0]
		elif type(proteinList) == str:
			protein = proteinList
		else:
			print('ERROR proteinList!')
			sys.exit()
		receptor, lig_native, listOfDecoys = Func_ReadPDB(proteinList, strProteinFolder, strTmpfs)
		# print('pass Func_ReadPDB') #debug
		if receptor is None:
			print('Error in %s receptor' % (protein))
			# file.write('Error in %s receptor\n' % (protein))
			# continue
			return (False, protein + '-receptor', strOutputFolder + '/ErrorProtein.txt')

		dictForHDF = {}
		hdf = h5py.File(strOutputFolder + '/' + protein + '.h5', 'w', libver='latest')
		for decoy in listOfDecoys:  # decoy:[decoy_name, decoy_mol]
			if decoy[1] is None:
				continue
			ProteinAtomFeatures, ProteinAtomsIdx = \
				Func_GetProteinFeatures(receptor, protein, strProteinFolder)
			LigandAtomFeatures = Func_GetLigandFeatures(decoy[1], protein, decoy[0])
			# shell-like mode
			decoyName = decoy[0].split('.pdb')[0]
			dictForHDF[decoyName] = {}
			dictForHDF[decoyName]['LigandAtomFeatures'] = LigandAtomFeatures   # n * 8 integer
			listOfShellDist = []
			listOfShellDist.append(0.0)
			for shell in range(ShellNum):
				listOfShellDist.append(D0+shell*GAP)
			# Used to check whether the current named decoy would get two connected subgraph
			# with Chem.Rdmolops.GetMolFrags
			decoy_mol = decoy[1]
			lig_list = Chem.rdmolops.GetMolFrags(mol=decoy_mol)
			if len(lig_list) > 1: # more than 1 connected subgraph
				ConnectedMatrix = Func_ConstructConnectedMatrix(lig_list, decoy_mol)
			else:
				ConnectedMatrix = Chem.rdmolops.GetAdjacencyMatrix(mol=decoy_mol)
			for i in range(len(listOfShellDist)-1):
				ComplexAdjMatrix, EdgeType, mapProteinAtomIdx2MatrixIndex, mapLigandAtomIdx2MatrixIndex = \
					Func_GetEdgeFeatures(ProteinAtomsIdx, decoy[1], ConnectedMatrix, listOfShellDist[i], listOfShellDist[i+1])
				AdjacencyDistMatrix, PathMatrix = Func_Floyd(ComplexAdjMatrix)
				# print('pass Func_GetEdgeFeatures(ProteinAtomsInBox, decoy[1])') #debug
				ProteinAtomFeatures_shell = {}
				for p_atomIdx in mapProteinAtomIdx2MatrixIndex:
					ProteinAtomFeatures_shell[p_atomIdx] = ProteinAtomFeatures[p_atomIdx]
				dictForHDF[decoyName][i+1] = {}
				dictForHDF[decoyName][i+1]['ProteinAtomFeatures'] = ProteinAtomFeatures_shell # m * 7 integer
				dictForHDF[decoyName][i+1]['AdjacencyMatrix'] = ComplexAdjMatrix     # (m+n) * (m+n) np.float32
				dictForHDF[decoyName][i+1]['AdjacencyDistMatrix'] = AdjacencyDistMatrix		  # (m+n) * (m+n)

				EdgeTypeForPath = np.zeros(shape=(ComplexAdjMatrix.shape[0], ComplexAdjMatrix.shape[1]))
				for k in EdgeType.keys():
					P, L = k.split('-')
					if P[0] == 'P':
						indexP = mapProteinAtomIdx2MatrixIndex[int(P[1:])]
					else:
						indexP = mapLigandAtomIdx2MatrixIndex[int(P[1:])]
					indexL = mapLigandAtomIdx2MatrixIndex[int(L[1:])]
					EdgeTypeForPath[indexP, indexL] = EdgeType[k]
					EdgeTypeForPath[indexL, indexP] = EdgeType[k]
				num_atom = PathMatrix.shape[0]
				edge_input = np.zeros([num_atom, num_atom, max_hop], dtype=np.int8)
				for BeginAtomIndex in range(num_atom):
					for EndAtomIndex in range(BeginAtomIndex, num_atom):
						if PathMatrix[BeginAtomIndex][EndAtomIndex] >= 0:  # 说明BeginAtomIndex-EndAtomIndex存在路径
							TransitionPoint = PathMatrix[BeginAtomIndex][EndAtomIndex]
							PathList = []
							PathList.append(EndAtomIndex)
							while TransitionPoint != BeginAtomIndex:
								PathList.append(TransitionPoint)
								TransitionPoint = PathMatrix[BeginAtomIndex][TransitionPoint]
							PathList.append(TransitionPoint)
							PathList.reverse()  # path from BeginAtomIndex to EndAtomIndex
							StartIndex = PathList[0]
							for count, TransitionIndex in enumerate(PathList[1 : max_hop + 1]):  # no more than max_hop
								edge_input[BeginAtomIndex, EndAtomIndex, count] = EdgeTypeForPath[StartIndex][TransitionIndex]
								StartIndex = TransitionIndex
							# Work with the lower triangular part of edge_input
							# , noting that the path is opposite to the upper triangular part
							PathList.reverse()
							StartIndex = PathList[0]
							for count, TransitionIndex in enumerate(PathList[1 : max_hop + 1]):
								edge_input[EndAtomIndex, BeginAtomIndex, count] = EdgeTypeForPath[StartIndex][TransitionIndex]
								StartIndex = TransitionIndex
				dictForHDF[decoyName][i+1]['edge_input'] = edge_input
		Func_DumpHDF(hdf, dictForHDF)
		hdf.close()
		return (True, protein, strOutputFolder + '/ErrorProtein.txt')
	except Exception as err:
		print('ERROR in %s:' % (protein))
		traceback.print_exc()


global_index = 0
global_total = 0
if __name__ == '__main__':
	params = argparser()
	strProteinFolder = params['F']
	strOutputFolder = params['O']
	PN = params['P']
	ShellNum = params['S']
	ListOfDecoys = params['list']
	max_hop = params['max_hop']
	# rec_dist = params['rec_dist']
	strTmpfs = params['tmpfs']
	GAP = params['gap']
	D0 = params['d0']

	if ListOfDecoys=='':
		ListOfDecoys = strProteinFolder + '/ProteinList.txt'

	if not os.path.exists(ListOfDecoys):
		print('ProteinList.txt not found in %s' % (strProteinFolder))
		sys.exit()

	file = open(ListOfDecoys)
	lines = file.readlines()
	file.close()

	ProteinList = []
	if len(lines[0].split()) > 1: #protein  Lig_*  RMSD
		pre_protein = ''
		tempList = []
		for l in lines:
			tempL = l.split()
			if tempL[0] != pre_protein:
				if tempList:
					ProteinList.append([pre_protein, tempList])
					tempList = []
				pre_protein = tempL[0]
			if tempL[1][:5] == 'Lig_0':
				tempList.append('Lig_native.pdb')
			else:
				tempList.append(tempL[1].split('.')[0]+'.pdb')
		if tempList:
			ProteinList.append([pre_protein, tempList])
	else:
		for p in lines:
			ProteinList.append(p.strip())

	if strOutputFolder == '':
		strOutputFolder = strProteinFolder + '_out'
	if not os.path.exists(strOutputFolder):
		os.mkdir(strOutputFolder)
	if not os.path.exists(strTmpfs):
		os.mkdir(strTmpfs)

	totalProtein = len(ProteinList)
	global_total = totalProtein
	# file = open(strOutputFolder+'/ErrorProtein.txt', 'w')
	process = multiprocessing.Pool(PN)
	for index, p in enumerate(ProteinList):
		process.apply_async(func=Func_MultiProcess
							, args=(p, strProteinFolder, strOutputFolder, max_hop, strTmpfs, ShellNum, GAP, D0,)
							, callback=Callback_WriteERRORProtein)
	process.close()
	process.join()

	if os.path.exists(strTmpfs):
		os.rmdir(strTmpfs)
	# file.close()
	print('\nFinished ALL!')
