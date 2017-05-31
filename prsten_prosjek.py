import Net
import numpy as np
import display_results
import matplotlib.pyplot as plt
inertion=[]
no_inertion=[]
for J in range(200):
    print("J:",J)
    net = Net.FeedForwardNet(input_count=2, layers=[5, 4, 2], activation_function=Net.FeedForwardNet.tanh)
    in_values, out_values = display_results.parse_data_file("prsten/prsten_in.npy","prsten/prsten_out.npy")
    # epoch_num=100000
    epoch_num=15000
    error=[]
    if(J % 2 == 0):
        I=None
    else:
        I=0.5
    for i in range(epoch_num):
        if i % 1000 == 0:
            print("Epoch:",i)
        batch_in,batch_out=net.generate_random_batch(in_values,out_values,60)

        # for start, end in zip(range(0, len(in_values), 60), range(60, len(in_values) + 1, 60)):
        #     net.forward_propagation(in_values[start:end])
        #     net.stochastic_backpropagation(out_values[start:end],learning_rate=0.0015,inertion_factor=0.3)

        net.forward_propagation(batch_in)
        net.stochastic_backpropagation(batch_out, learning_rate=0.005, inertion_factor=I)
        # if net.check_maximum_error(batch_out, 0.1, verbose=True,error_list=error):
        #     break
        if net.check_total_squared_error(batch_out,1, verbose=False,error_list=error):
            break


    # display_results.plot_error(error)
    # display_results.display_results(net,in_values,out_values)
    if(J % 2 == 0):
        # no intertion
        no_inertion.append(len(error))
    else:
        # inertion
        inertion.append(len(error))

no_iner_avg=np.average(np.array(no_inertion))
iner_avg=np.average(np.array(inertion))
print("noiner:",no_iner_avg)
print("iner:",iner_avg)
