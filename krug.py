from tkinter import *
import Net
import numpy as np
import display_results

net = Net.FeedForwardNet(input_count=2, layers=[5, 2], activation_function=Net.FeedForwardNet.tanh)
in_values, out_values = display_results.parse_data_file("krug/circle_in.npy","krug/circle_out.npy")
epoch_num=100000
error=[]
for i in range(epoch_num):
    if i % 100 == 0:
        print("Epoch:",i)
    net.forward_propagation(in_values)
    net.backpropagation(out_values, learning_rate=0.05, inertion_factor=0.3)
    # net.stochastic_backpropagation(out_values, learning_rate=0.02)
    # if net.check_maximum_error(out_values, 1):
    #     break
    if net.check_maximum_error(out_values, 0.5, verbose=True,error_list=error):
        break

display_results.plot_error(error)
display_results.display_results(net,in_values,out_values)
net.save_state("krug/krug.p")