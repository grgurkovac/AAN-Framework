import Net
import numpy as np
import display_results
import matplotlib.pyplot as plt
net = Net.FeedForwardNet(input_count=2, layers=[10, 10, 10, 8, 2], activation_function=Net.FeedForwardNet.tanh)
in_values, out_values = display_results.parse_data_file("prsten/prsten_in.npy","prsten/prsten_out.npy")
# epoch_num=100000
epoch_num=15000
error=[]
for i in range(epoch_num):
    if i % 1000 == 0:
        print("Epoch:",i)
    batch_in,batch_out=net.generate_random_batch(in_values,out_values,60)

    # for start, end in zip(range(0, len(in_values), 60), range(60, len(in_values) + 1, 60)):
    #     net.forward_propagation(in_values[start:end])
    #     net.stochastic_backpropagation(out_values[start:end],learning_rate=0.0015,inertion_factor=0.3)

    net.forward_propagation(batch_in)
    net.backpropagation(batch_out, learning_rate=0.001, inertion_factor=0.3)
    # if net.check_maximum_error(batch_out, 0.1, verbose=True,error_list=error):
    #     break
    if net.check_total_squared_error(batch_out,0.1, verbose=True,error_list=error):
        break


display_results.display_results(net,in_values,out_values)
display_results.plot_error(error)

net.save_state("prsten/prsten.p")
