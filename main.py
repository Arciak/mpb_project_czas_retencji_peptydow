import numpy as np
import matplotlib.pyplot as plt
import readpeptides as rp
import savereadsynapticweights as srsw
from multilayeredneuralnetwork import NeuronLayer, NeuralNetwork



def korelacja(x, y):
	if x>y:
		return y/x
	else:
		return x/y

# Seed the random number generator
np.random.seed(1)

# Create layer 1 (2 neurons, each with 20 inputs)
layer1 = NeuronLayer(2, 21)

# Create layer 2 (1 neuron with 3 inputs)
layer2 = NeuronLayer(1, 3)


# Combine the layers to create a neural network
neural_network = NeuralNetwork(layer1, layer2)

#check_data = rp.read_data_from_file("rt_pred_data/krokhin.txt")

loop_bool = True
number_of_examples = 150
file_path = "rt_pred_data/krokhin.txt"
training_loops = 150

while loop_bool:

	loop_int = int(input(" \n 1-- ucz siec nowymi danymi \n 2-- przewiduj z zapisanymi wagami \n 3-- przewiduj i drukuj wykres \n 4-- zakoncz"))

	if loop_int == 1:
		array_mse = []
		mse22 = []

		print("Stage 1) Random starting synaptic weights: ")
		print(neural_network.print_weights())

		# The training set. We have  number_of_examples, each consisting of 20 input values
		# and 1 output value.
		# training_data = rp.read_data_from_file(file_path)
		# training_inputs = training_data[0][:number_of_examples]
		# training_outputs = np.array(training_data[1][:number_of_examples])
		#
		# # Train the neural network using the training set.
		# mse22 = neural_network.train(training_inputs, training_outputs, training_loops)
		#
		# print("\n Stage 2) New synaptic weights after training: ")
		# print(neural_network.print_weights())
		#
		# srsw.save_synaptic_weights(neural_network.weights())
		#
		for xx in range(number_of_examples):
			print("Numer danych: {}\n".format(xx))
			print("Stage 1) Random starting synaptic weights: ")
			print(neural_network.print_weights())

			# The training set. We have  number_of_examples, each consisting of 20 input values
			# and 1 output value.
			training_data = rp.read_data_from_file(file_path)
			training_inputs = training_data[0][xx:number_of_examples]
			training_outputs = np.array(training_data[1][xx:number_of_examples])

			# Train the neural network using the training set.
			mse22 = neural_network.train(training_inputs, training_outputs, training_loops)

			print("\n Stage 2) New synaptic weights after training: ")
			print(neural_network.print_weights())

			srsw.save_synaptic_weights(neural_network.weights())

			check_data = rp.read_data_from_file(file_path)

			suma_dla_mse = 0
			check_data_train = rp.read_data_from_file(file_path)
			real_data_train = check_data_train[1][number_of_examples:np.size(check_data_train[1][:,-1])]
			pred_out_train = []
			for iter, peptid in enumerate(check_data_train[0][number_of_examples:np.size(check_data_train[1][:,-1])]):
				temp_data_train = neural_network.think(np.array([peptid]))
				pred_out_train = np.append(pred_out_train, temp_data_train[1])
				diff = real_data_train[iter] - temp_data_train[1]
				pred_out_train = np.append(pred_out_train, temp_data_train[1])
				squered_diff = diff**2
				suma_dla_mse += squered_diff
			# MSE = np.square(np.subtract(check_data[1][number_of_examples:np.size(check_data[1])], pred_out)).mean()
			# print(MSE)
			real_data_train = check_data_train[1][number_of_examples:np.size(check_data_train[1][:,-1])]
			pred_data_train = pred_out_train
			temp_mse = suma_dla_mse/np.size(check_data_train[1][number_of_examples:np.size(check_data_train[1][:,-1])])
			array_mse = np.append(array_mse, temp_mse)

		print("Maluje wykres")
		plt.figure()
		plt.plot(array_mse)
		plt.xlabel("Liczba danch uczących")
		plt.ylabel("MSE")
		plt.title("Zależność od liczby danych uczących dla krokhin.txt")
		plt.grid()
		plt.show()

		plt.figure()
		plt.figure()
		plt.plot(mse22)
		plt.xlabel("Numer kroku")
		plt.ylabel("MSE")
		plt.title("Proces uczenia dla krokhin.txt")
		plt.grid()
		plt.show()



	elif loop_int == 2:
		# Test the neural network with a new situation.
		check_data = rp.read_data_from_file(file_path)
		neural_network.set_weights(srsw.read_synaptic_weights())
		while loop_bool:
			pred_loop_int = int(input(" \n 1-- inne dane \n 2-- zakoncz \n"))
			if pred_loop_int == 2:
				loop_bool = False
				break

			chosen_data = int(input("Podaj numer wiersza: "))
			data = check_data[0][chosen_data-1]
			print(" \n New situation input data. {} row form {}: \n {}" .format(chosen_data, file_path, data))
			print("Output Predicted time: ")
			print(neural_network.think(np.array([data])))

	elif loop_int == 3:
		check_data_train = rp.read_data_from_file(file_path)
		neural_network.set_weights(srsw.read_synaptic_weights())
		print(np.size(check_data_train[1]))
		pred_out_train = []
		for peptid in check_data_train[0][number_of_examples:np.size(check_data_train[1][:,-1])]:
			temp_data_train = neural_network.think(np.array([peptid]))
			pred_out_train = np.append(pred_out_train, temp_data_train[1])

		# MSE = np.square(np.subtract(check_data[1][number_of_examples:np.size(check_data[1])], pred_out)).mean()
		# print(MSE)
		real_data_train = check_data_train[1][number_of_examples:np.size(check_data_train[1][:,-1])]
		pred_data_train = pred_out_train
		korelacja_value_train = 0
		for counter, val in enumerate(real_data_train):
			korelacja_value_train += korelacja(val, pred_data_train[counter])

		korelacja_value_train = korelacja_value_train/np.size(pred_data_train)
		print(korelacja_value_train)
		plt.plot(real_data_train, real_data_train, color='blue')
		plt.plot(real_data_train, pred_out_train, 'ro', color = 'red')
		plt.grid()
		plt.title("Czas retencji peptydów dla krokhin.txtx")
		plt.xlabel("Realne [s]")
		plt.ylabel("Przewidziane [s]")
		plt.show()

	elif loop_int == 4:
		break


# wspolczynnik korelacji
# opisac procedure testowa
# parametry  sieci