import numpy as np
from tkinter import *
import matplotlib.pyplot as plt

def parse_data_file(filename_in,filename_out):
    """
    Loads input and output data using numpy and return it as numpy matrices.
    :param filename_in: file containing inputs
    :param filename_out: fie containing wanted outputs
    :return: two numpy matrices
    """
    input_matrix = np.load(filename_in)
    output_matrix = np.load(filename_out)
    return input_matrix, output_matrix

def display_results(net,in_values,out_values, talk=False, min=-1, max=1):
    def exit(event=None):
        master.destroy()

    # calculates the cartesian product
    domain = np.dstack(np.meshgrid(np.linspace(0, 1, 301), np.linspace(0, 1, 201))).reshape(-1, 2)
    net.X = [None] * net.layers_count  # represents input
    net.IN = [None] * net.layers_count  # represents input after we sum it and apply the weights
    net.Y = [None] * net.layers_count  # represents output of the activation function
    Y = net.forward_propagation(domain)

    master = Tk()
    master.title("Plot")
    w = Canvas(master, width=300, height=200)
    w.focus_force()
    w.bind("<space>", exit)
    for dot in zip(domain, Y):
        red, blue = dot[1]
        x, y = dot[0]
        red += 1
        blue += 1
        total = red + blue
        red_factor = red / total
        blue_factor = blue / total
        color = '#%02x%02x%02x' % (int(red_factor * 200), int(250), int(blue_factor * 200))

        if talk and x == y:
            print(red_factor, ":", blue_factor)

        w.create_line(x * 300, y * 200, x * 300, y * 200, fill=color)

    for dot in zip(in_values, out_values):
        if dot[1][1] == max:
            color = 'green'
        elif dot[1][1] == min:
            color = 'red'
        x1, y1 = (dot[0][0] * 300 - 1), (dot[0][1] * 200 - 1)
        x2, y2 = (dot[0][0] * 300 + 1), (dot[0][1] * 200 + 1)
        w.create_oval(x1, y1, x2, y2, fill=color, outline=color)
    w.pack()
    mainloop()

def plot_error(y,x=None):
    if x is None:
        x = range(len(y))
    plt.scatter(range(len(y)), y,s=0.1)
    plt.show()
