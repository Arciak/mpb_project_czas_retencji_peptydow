################################################################################
# training_inputs ==> tablica 2D. Kolumny to odpowiednie 20 rodzajow peptydow. #
# Wiersze prezentuja odpowiednia sekwencje peptydow przeczytana z pliku.       #
# training_outputs ==> tablica 1D. Pezentuje czasu retencji przeczytanych      #
# sekwencji peptydow                                                           #
################################################################################

import numpy as np


def read_data_from_file(file_path):
	'''
	
	:param file_path:
	:return:
	'''
	read_peptides = []
	training_outputs = []

	with open(file_path, 'r') as peptideFile:
		for lineInFile in peptideFile:
			line = lineInFile.split('\t')
			read_peptides = np.append(read_peptides, line[0])
			training_outputs = np.append(training_outputs, float(line[1]))

	letters = []

	for peptide in read_peptides:
		for letter in peptide:
			if letter in letters:
				continue
			else:
				letters = np.append(letters, letter)

	letters = np.sort(letters)

	training_inputs = np.zeros((np.size(read_peptides), np.size(letters)))

	temp_iterator = -1
	for peptide in read_peptides:
		temp_iterator += 1
		for letter in peptide:
			training_inputs[temp_iterator, np.where(letters == letter)] += 1

	training_outputs.shape = (training_outputs.size, 1)
	return training_inputs, training_outputs




