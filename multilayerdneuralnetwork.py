from numpy import exp, array, random, dot, size
import readpeptides as rp
import savereadsynapticweights as srsw


class NeuronLayer():
	def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
		self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
	def __init__(self, layer1, layer2):
		self.layer1 = layer1
		self.layer2 = layer2

	def weights(self):
		return self.layer1.synaptic_weights, self.layer2.synaptic_weights

	# The Sigmoid function, which describes an S shaped curve.
	# We pass the weighted sum of the inputs through this function to
	# normalise them between 0 and 1.
	def __sigmoid(self, x):
		return 1 / (1 + exp(-x))

	# The derivative of the Sigmoid function.
	# This is the gradient of the Sigmoid curve.
	# It indicates how confident we are about the existing weight.
	def __sigmoid_derivative(self, x):
		return x * (1 - x)

	# We train the neural network through a process of trial and error.
	# Adjusting the synaptic weights each time.
	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
		for iteration in range(number_of_training_iterations):
			# Pass the training set through our neural network
			output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

			# Calculate the error for layer 2 (The difference between the desired output
			# and the predicted output).
			layer2_error = training_set_outputs - output_from_layer_2
			layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

			# Calculate the error for layer 1 (By looking at the weights in layer 1,
			# we can determine by how much layer 1 contributed to the error in layer 2).
			layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
			layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

			# Calculate how much to adjust the weights by
			layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
			layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

			# Adjust the weights.
			self.layer1.synaptic_weights += layer1_adjustment
			self.layer2.synaptic_weights += layer2_adjustment

	# The neural network thinks.
	def think(self, inputs):
		output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
		output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
		return output_from_layer1, output_from_layer2

	# The neural network prints its weights
	def print_weights(self):
		print ("    Layer 1 (4 neurons, each with 3 inputs): ")
		print (self.layer1.synaptic_weights)
		print ("    Layer 2 (1 neuron, with 4 inputs):")
		print (self.layer2.synaptic_weights)

#if __name__ == "__main__":
########################################## my example #############################################################

#Seed the random number generator
random.seed(1)

# Create layer 1 (4 neurons, each with 3 inputs)
layer1 = NeuronLayer(40, 20)

# Create layer 2 (a single neuron with 4 inputs)
layer2 = NeuronLayer(1, 40)

# Combine the layers to create a neural network
neural_network = NeuralNetwork(layer1, layer2)

print("Stage 1) Random starting synaptic weights: ")
print(neural_network.print_weights())

# The training set. We have  number_of_examples, each consisting of 20 input values
# and 1 output value.
number_of_examples = 230
training_data = rp.read_data_from_file("rt_pred_data/krokhin.txt")
training_inputs = training_data[0][:number_of_examples]
training_outputs = array(training_data[1][:number_of_examples])

# Train the neural network using the training set.
# Do it 60,000 times and make small adjustments each time.
neural_network.train(training_inputs, training_outputs, 50000)

print("Stage 2) New synaptic weights after training: ")
print(neural_network.print_weights())

# Test the neural network with a new situation.
loop_bool = True
check_data = rp.read_data_from_file("rt_pred_data/mouse.txt")

while loop_bool:
	loop_int = int(input(" \n 1-- inne dane \n 2-- zakoncz \n"))
	if loop_int == 2:
		loop_bool = False
		break

	chosen_data = int(input("Podaj numer wiersza: "))
	data = check_data[0][chosen_data-1]
	print(" \n New situation input data. {} row form krokhin.txt: \n {}" .format(chosen_data, data))
	print("Output Predicted time: ")
	print(neural_network.think(array([data])))

srsw.save_synaptic_weights(neural_network.weights())


# #Seed the random number generator
# random.seed(1)
#
# # Create layer 1 (4 neurons, each with 3 inputs)
# layer1 = NeuronLayer(4, 3)
#
# # Create layer 2 (a single neuron with 4 inputs)
# layer2 = NeuronLayer(1, 4)
#
# # Combine the layers to create a neural network
# neural_network = NeuralNetwork(layer1, layer2)
#
# print ("Stage 1) Random starting synaptic weights: ")
# print(neural_network.print_weights())
#
# # The training set. We have 7 examples, each consisting of 3 input values
# # and 1 output value.
# training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
# training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T
#
# # Train the neural network using the training set.
# # Do it 60,000 times and make small adjustments each time.
# neural_network.train(training_set_inputs, training_set_outputs, 60000)
#
# print( "Stage 2) New synaptic weights after training: ")
# print(neural_network.print_weights())
#
# # Test the neural network with a new situation.
# print("Stage 3) Considering a new situation [1, 1, 0] -> ?: ")
# hidden_state, output = neural_network.think(array([1, 1, 0]))
# print (output)




