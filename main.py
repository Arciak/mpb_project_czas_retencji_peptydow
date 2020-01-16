from numpy import exp, array, random, dot, size
from time import time as my_timer
import readpeptides as rp
import savereadsynapticweights as srsw
from multilayeredneuralnetwork import NeuronLayer, NeuralNetwork


# Seed the random number generator
random.seed(1)

# Create layer 1 (2 neurons, each with 20 inputs)
layer1 = NeuronLayer(2, 20)

# Create layer 2 (1 neuron with 2 inputs)
layer2 = NeuronLayer(1, 2)


# Combine the layers to create a neural network
neural_network = NeuralNetwork(layer1, layer2)

check_data = rp.read_data_from_file("rt_pred_data/krokhin.txt")

loop_bool = True

while loop_bool:
	loop_int = int(input(" \n 1-- ucz siec nowymi danymi \n 2-- przewiduj z zapisanymi wagami \n"))
	if loop_int == 1:

		print("Stage 1) Random starting synaptic weights: ")
		print(neural_network.print_weights())

		# The training set. We have  number_of_examples, each consisting of 20 input values
		# and 1 output value.
		number_of_examples = 205
		training_data = rp.read_data_from_file("rt_pred_data/petritis.txt")
		training_inputs = training_data[0][:number_of_examples]
		training_outputs = array(training_data[1][:number_of_examples])

		# Train the neural network using the training set.
		neural_network.train(training_inputs, training_outputs, 591000)

		print("\n Stage 2) New synaptic weights after training: ")
		print(neural_network.print_weights())

		srsw.save_synaptic_weights(neural_network.weights())

	elif loop_int == 2:
		# Test the neural network with a new situation.
		check_data = rp.read_data_from_file("rt_pred_data/petritis.txt")
		neural_network.set_weights(srsw.read_synaptic_weights())
		while loop_bool:
			pred_loop_int = int(input(" \n 1-- inne dane \n 2-- zakoncz \n"))
			if pred_loop_int == 2:
				loop_bool = False
				break

			chosen_data = int(input("Podaj numer wiersza: "))
			data = check_data[0][chosen_data-1]
			print(" \n New situation input data. {} row form petritis.txt: \n {}" .format(chosen_data, data))
			print("Output Predicted time: ")
			print(neural_network.think(array([data])))


# wspolczynnik korelacji
# opisac procedure testowa
# parametry  sieci