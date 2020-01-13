from numpy import exp, array, random, dot, size
from time import time as my_timer
import readpeptides as rp
import savereadsynapticweights as srsw


class NeuronLayer():
	def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
		self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
	def __init__(self, layer1, layer2, layer3):
		self.layer1 = layer1
		self.layer2 = layer2
		self.layer3 = layer3

	def weights(self):
		return self.layer1.synaptic_weights, self.layer2.synaptic_weights, self.layer3.synaptic_weights

	def set_weights(self, weights):
		self.layer1.synaptic_weights = weights[0]
		self.layer2.synaptic_weights = weights[1]
		self.layer3.synaptic_weights = weights[2]

	# The Sigmoid function, which describes an S shaped curve.
	# We pass the weighted sum of the inputs through this function to
	# normalise them between 0 and 1.
	def __sigmoidal_function(self, x):
		return 1 / (1 + exp(-x))

	# The derivative of the Sigmoid function.
	# This is the gradient of the Sigmoid curve.
	# It indicates how confident we are about the existing weight.
	def __sigmoidal_derivative(self, x):
		return x * (1 - x)

	# We train the neural network through a process of trial and error.
	# Adjusting the synaptic weights each time.
	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
		forloop_int = 0
		start_time = my_timer()
		for iteration in range(number_of_training_iterations):
			++forloop_int

			# Pass the training set through our neural network
			output_from_layer_1, output_from_layer_2, output_from_layer_3 = self.think(training_set_inputs)

			# Calculate the error for layer 3 (The difference between the desired output
			# and the predicted output).
			layer3_error = training_set_outputs - output_from_layer_3
			layer3_delta = layer3_error * self.__sigmoidal_derivative(output_from_layer_3)

			# Calculate the error for layer 2 (By looking at the weights in layer 2,
			# we can determine by how much layer 2 contributed to the error in layer 3).
			layer2_error = layer3_delta.dot(self.layer3.synaptic_weights.T)
			layer2_delta = layer2_error * self.__sigmoidal_derivative(output_from_layer_2)

			# Calculate the error for layer 1 (By looking at the weights in layer 1,
			# we can determine by how much layer 1 contributed to the error in layer 2).
			layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
			layer1_delta = layer1_error * self.__sigmoidal_derivative(output_from_layer_1)

			# Calculate how much to adjust the weights by
			layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
			layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)
			layer3_adjustment = output_from_layer_2.T.dot(layer3_delta)

			# Adjust the weights.
			self.layer1.synaptic_weights += layer1_adjustment
			self.layer2.synaptic_weights += layer2_adjustment
			self.layer3.synaptic_weights += layer3_adjustment

			if forloop_int/10000000 in range(1, 91):
				end_time = my_timer()
				print("Krok {} z {}".format(forloop_int/10000000, number_of_training_iterations/10000000))
				print("Czas trwania obliczen: {} \n".format(end_time - start_time))

	# The neural network thinks.
	def think(self, inputs):
		output_from_layer1 = self.__sigmoidal_function(dot(inputs, self.layer1.synaptic_weights))
		output_from_layer2 = self.__sigmoidal_function(dot(output_from_layer1, self.layer2.synaptic_weights))
		output_from_layer3 = self.__sigmoidal_function(dot(output_from_layer2, self.layer3.synaptic_weights))
		return output_from_layer1, output_from_layer2, output_from_layer3

	# The neural network prints its weights
	def print_weights(self):
		print("    Layer 1 (20 neurons, each with 20 inputs): ")
		print(self.layer1.synaptic_weights)
		print("    Layer 2 (5 neuron, with 20 inputs):")
		print(self.layer2.synaptic_weights)
		print("    Layer 3 (1 neuron, with 5 inputs):")
		print(self.layer3.synaptic_weights)


if __name__ == "__main__":

	# Seed the random number generator
	random.seed(1)

	# Create layer 1 (20 neurons, each with 20 inputs)
	layer1 = NeuronLayer(20, 20)

	# Create layer 2 (5 neurons with 20 inputs)
	layer2 = NeuronLayer(5, 20)

	# Create layer 3 (1 neurons with 5 inputs)
	layer3 = NeuronLayer(1, 5)

	# Combine the layers to create a neural network
	neural_network = NeuralNetwork(layer1, layer2, layer3)

	check_data = rp.read_data_from_file("rt_pred_data/krokhin.txt")

	loop_bool = True

	while loop_bool:
		loop_int = int(input(" \n 1-- ucz siec nowymi danymi \n 2-- przewiduj z zapisanymi wagami \n"))
		if loop_int == 1:

			print("Stage 1) Random starting synaptic weights: ")
			print(neural_network.print_weights())

			# The training set. We have  number_of_examples, each consisting of 20 input values
			# and 1 output value.
			number_of_examples = 229
			training_data = rp.read_data_from_file("rt_pred_data/krokhin.txt")
			training_inputs = training_data[0][:number_of_examples]
			training_outputs = array(training_data[1][:number_of_examples])

			# Train the neural network using the training set.
			neural_network.train(training_inputs, training_outputs, 10000000)

			print("Stage 2) New synaptic weights after training: ")
			print(neural_network.print_weights())

			srsw.save_synaptic_weights(neural_network.weights())

		elif loop_int == 2:
			# Test the neural network with a new situation.
			check_data = rp.read_data_from_file("rt_pred_data/krokhin.txt")
			neural_network.set_weights(srsw.read_synaptic_weights())
			while loop_bool:
				pred_loop_int = int(input(" \n 1-- inne dane \n 2-- zakoncz \n"))
				if pred_loop_int == 2:
					loop_bool = False
					break

				chosen_data = int(input("Podaj numer wiersza: "))
				data = check_data[0][chosen_data-1]
				print(" \n New situation input data. {} row form mouse.txt: \n {}" .format(chosen_data, data))
				print("Output Predicted time: ")
				print(neural_network.think(array([data])))
