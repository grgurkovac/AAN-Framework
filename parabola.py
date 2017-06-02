import Net
import numpy as np
import display_results
net = Net.FeedForwardNet(input_count=2, layers=[3, 2], activation_function=Net.FeedForwardNet.tanh)
in_values, out_values = display_results.parse_data_file("parabola/parabola_in.npy","parabola/parabola_out.npy")
epoch_num=15000
errors=[]
for i in range(epoch_num):
    if i % 100 ==0:
        print("Epoch:",i)
    net.forward_propagation(in_values)
    # net.stochastic_backpropagation(out_values, learning_rate=0.02)
    net.backpropagation(out_values, learning_rate=0.005)
    # net.stochastic_backpropagation(out_values, learning_rate=0.08)

    # if net.check_maximum_error(out_values, 0.2, verbose=True,error_list=errors):
    #     break
    if net.check_total_squared_error(output_values=out_values, epsilon=2, verbose=True, error_list=errors):
        break

display_results.plot_error(errors)
display_results.display_results(net,in_values,out_values)
net.save_state("parabola/parabola.p")
