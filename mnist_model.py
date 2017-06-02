import Net
import mnist_parser
import numpy as np
#To use this model it is required to download the MNIST database
#The donwloaded base is then needet parse to numpy using mnist_parser.parse_to_npy method
#The files genetared using mnist_parser.parse_to_npy are then loaded using np.load
in_values = np.load("MNIST/mnist_train_images.npy")
out_values = np.load("MNIST/mnist_train_labels.npy")
out_gt_numbers=mnist_parser.one_hots_to_ints(out_values)

in_testing_values = np.load("MNIST/mnist_test_images.npy")
out_testing_values = np.load("MNIST/mnist_test_labels.npy")
out_gt_numbers_test=mnist_parser.one_hots_to_ints(out_testing_values)

net = Net.FeedForwardNet(input_count=784, layers=[100, 10], activation_function=Net.FeedForwardNet.leaky_relu)

epoch_num=10000
batch_size=40
learning_rate=0.001
inertion_factor=0.3

for i in range(epoch_num):
    batch_in,batch_out=net.generate_random_batch(in_values,out_values,batch_size)
    net.forward_propagation(batch_in)
    net.backpropagation(batch_out, learning_rate=learning_rate, inertion_factor=inertion_factor)

    if i % 50 == 0:
        print()
        output=net.forward_propagation(in_testing_values)
        if net.check_total_squared_error(output_values=out_testing_values, epsilon=300, verbose=True):
            break
        output_numbers=mnist_parser.one_hots_to_ints(output)
        correct=np.sum(out_gt_numbers_test == output_numbers)
        print("Epoch: ", i, " br tocnih:",correct,"/",output_numbers.size,"(",correct/output_numbers.size,"%)")


output=net.forward_propagation(in_testing_values)
conf_mat=net.calculate_confusion_matrix(out_testing_values)

output_numbers = mnist_parser.one_hots_to_ints(output)
correct=np.sum(out_gt_numbers_test == output_numbers)
print("Correct:",correct,"/",output_numbers.size,"(",correct/output_numbers.size ,"%)")

print(conf_mat)
net.save_state("MNIST/mnist.p")
