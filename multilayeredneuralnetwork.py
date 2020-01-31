import numpy as np
import matplotlib.pyplot as plt


class NeuronLayer():
	def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
		self.synaptic_weights = 2 * np.random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
	def __init__(self, layer1, layer2):
		self.layer1 = layer1
		self.layer2 = layer2

	def weights(self):
		return self.layer1.synaptic_weights, self.layer2.synaptic_weights

	def set_weights(self, weights):
		self.layer1.synaptic_weights = weights[0]
		self.layer2.synaptic_weights = weights[1]

	def __sigmoidal_function(self, x):
		return 1 / (1 + np.exp(-x))

	def __linear_function(self, x):
		return 0.01*x

	def __sigmoidal_derivative(self, x):
		return x * (1-x)

	def __linear_derivative(self, x):
		return 0.01

	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
		array_mse = []
		for iteration in range(number_of_training_iterations):

			# Pass the training set through our neural network
			output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

			temp_mse = np.square(np.subtract(training_set_outputs, output_from_layer_2)).mean()
			array_mse = np.append(array_mse, temp_mse)

			# Calculate the error for layer 2 (By looking at the weights in layer 2,
			# we can determine by how much layer 2 contributed to the error in layer 3).
			layer2_error = training_set_outputs - output_from_layer_2
			layer2_delta = layer2_error * self.__linear_derivative(output_from_layer_2)


			# Calculate the error for layer 1 (By looking at the weights in layer 1,
			# we can determine by how much layer 1 contributed to the error in layer 2).

			#temp_layer2_synaptic_weights = np.delete(self.layer2.synaptic_weights, -1, 0)
			layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
			temp_layer1_error = np.delete(layer1_error, 0, -1)
			layer1_delta = temp_layer1_error * self.__sigmoidal_derivative(output_from_layer_1)

			# Calculate how much to adjust the weights by
			layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
			temp_output_from_layer_1 = np.c_[output_from_layer_1, np.ones(np.size(output_from_layer_1[:, -1]))]

			layer2_adjustment = temp_output_from_layer_1.T.dot(layer2_delta)

			# Adjust the weights.
			self.layer1.synaptic_weights += layer1_adjustment
			self.layer2.synaptic_weights += layer2_adjustment
		# print("Maluje wykres")
		# plt.plot(array_mse)
		# plt.xlabel("Numer kroku")
		# plt.ylabel("MSE")
		# plt.title("MSE")
		# plt.grid()
		# plt.show()

	# The neural network thinks.
	def think(self, inputs):
		output_from_layer1 = self.__sigmoidal_function(np.dot(inputs, self.layer1.synaptic_weights))
		#output_from_layer1[:, -1] = np.ones(np.size(output_from_layer1[:, 0]))
		temp_output_from_layer1 = np.c_[output_from_layer1, np.ones(np.size(output_from_layer1[:, -1]))]
		output_from_layer2 = self.__linear_function(np.dot(temp_output_from_layer1, self.layer2.synaptic_weights))
		return output_from_layer1, output_from_layer2

	# The neural network prints its weights
	def print_weights(self):
		print("    Layer 1 (2 neurons, each with 20 inputs): ")
		print(self.layer1.synaptic_weights)
		print("    Layer 2 (1 neuron, with 2 inputs):")
		print(self.layer2.synaptic_weights)



# wspolczynnik korelacji
# opisac procedure testowa
# parametry  sieci