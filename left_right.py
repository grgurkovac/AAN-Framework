import Net
import display_results
import numpy as np
import pickle
in_values, out_values = display_results.parse_data_file("left_right/left_right_in.npy","left_right/left_right_out.npy")
net = Net.FeedForwardNet(input_count=2, layers=[2], activation_function=Net.FeedForwardNet.tanh)
epoch_num=10
for i in range(epoch_num):
    net.forward_propagation(in_values)
    net.stochastic_backpropagation(out_values, learning_rate=0.1, inertion_factor=0.001)
display_results.display_results(net,in_values,out_values)
net.save_state("left_right/lr.p")
