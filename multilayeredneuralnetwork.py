from numpy import exp, array, random, dot, size


class NeuronLayer():
	def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
		self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


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
		return 1 / (1 + exp(-x))

	def __sigmoidal_derivative(self, x):
		return x * (1 - x)

	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
		# forloop_int = 0
		# divider = 1000
		# start_time = my_timer()
		for iteration in range(number_of_training_iterations):

			# Pass the training set through our neural network
			output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

			# Calculate the error for layer 3 (The difference between the desired output
			# and the predicted output).

			# Calculate the error for layer 2 (By looking at the weights in layer 2,
			# we can determine by how much layer 2 contributed to the error in layer 3).
			layer2_error = training_set_outputs - output_from_layer_2
			layer2_delta = layer2_error * self.__sigmoidal_derivative(output_from_layer_2)

			# Calculate the error for layer 1 (By looking at the weights in layer 1,
			# we can determine by how much layer 1 contributed to the error in layer 2).
			layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
			layer1_delta = layer1_error * self.__sigmoidal_derivative(output_from_layer_1)

			# Calculate how much to adjust the weights by
			layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
			layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

			# Adjust the weights.
			self.layer1.synaptic_weights += layer1_adjustment
			self.layer2.synaptic_weights += layer2_adjustment

			# forloop_int += 1
			#
			# if forloop_int % divider == 0 and int(forloop_int/divider) in range(1, 91):
			# 	end_time = my_timer()
			# 	print("Krok {} z {}".format(forloop_int/divider, number_of_training_iterations/divider))
			# 	print("Czas trwania obliczen: {} min \n".format(float((end_time - start_time)/60)))

	# The neural network thinks.
	def think(self, inputs):
		output_from_layer1 = self.__sigmoidal_function(dot(inputs, self.layer1.synaptic_weights))
		output_from_layer2 = self.__sigmoidal_function(dot(output_from_layer1, self.layer2.synaptic_weights))
		return output_from_layer1, output_from_layer2

	# The neural network prints its weights
	def print_weights(self):
		print("    Layer 1 (2 neurons, each with 20 inputs): ")
		print(self.layer1.synaptic_weights)
		print("    Layer 2 (1 neuron, with 2 inputs):")
		print(self.layer2.synaptic_weights)