
import sys, os, argparse, traceback
import h5py


def argparser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-F', type=str, required=True, help='the HDF folder')
	parser.add_argument('-O', type=str, default='output_reduce.hdf5', help='the output file name (default: output.hdf5)')
	parser.add_argument('-L', type=str, required=True, help='the Pose list file')
	args = parser.parse_args()
	params = vars(args)
	return params


if __name__ == '__main__':
	params = argparser()
	strHDFFolder = params['F']
	strOutput = params['O']
	strList = params['L']

	file = open(strList)
	lines = file.readlines()
	file.close()

	PoseList = []
	for l in lines:
		PoseList.append(l.strip())
	ProteinDict = {}
	prePDBName = ''
	tempList = []
	for p in PoseList:
		tempP = p.split()
		currentPDBName = tempP[0]
		if prePDBName != currentPDBName:
			if tempList:
				ProteinDict[prePDBName] = tempList
			prePDBName = currentPDBName
			tempList = []
		tempList.append(tempP[1])

	ProteinDict[prePDBName] = tempList

	# hdf5 = {}
	HDF5File = h5py.File(strOutput, 'w', libver='latest')
	total = len(PoseList)
	count = 0
	for protein in ProteinDict.keys():
		if not os.path.exists(strHDFFolder+'/'+protein+'.h5'):
			print('%s not found!' % (strHDFFolder+'/'+protein+'.h5'))
			continue
		with h5py.File(strHDFFolder+'/'+protein+'.h5', 'r') as f:
			group = HDF5File.create_group(''.join(['/',protein,'/']))
			for pose in ProteinDict[protein]:
				try:
					if pose=='Lig_native' or pose=='Lig_0':
						if 'Lig_native' in f:
							MyDebug = 'Lig_native'
						else:
							MyDebug = 'Lig_0'
					else:
						MyDebug = pose
					g = f.require_group(MyDebug)
					# LIG = group.create_group(''.join(['/',protein, '/', pose]))
					f.copy(source=g, dest=group)
					count += 1
					# print('\rFinished %d/%d ...' % (count, total), end='', flush=True)
				except Exception as err:
					print('ERROR in %s' % (protein+'/'+pose))
					traceback.print_exc()

	# deepdish.io.save(strOutput, hdf5, compression=('blosc', 5))
	HDF5File.close()
	print('Finished ALL!')
