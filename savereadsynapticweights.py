import numpy as np


def save_synaptic_weights(synaptic_weights):
	number_of_layers = np.size(synaptic_weights)

	for layer in range(number_of_layers):
		with open("layer_{}_weights.txt".format(layer), 'w') as weights:
			np.savetxt(weights, synaptic_weights[layer])  #zapis macierzy

	print("Zapisaono wagi do plikow")