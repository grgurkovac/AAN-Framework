from tkinter import *
import numpy as np

WIDTH = 300
HEIGHT = 200
master = Tk()
master.title("Drawing tool")
w = Canvas(master, width=WIDTH, height=HEIGHT)
group = IntVar()
color = StringVar()
cleared = BooleanVar()
cleared.set(False)
color.set('red')
filename = StringVar()
filename.set("filename")
dots =[]
groups = []

def toogleGroup(event=None):
    group.set(1 - group.get())
    if group.get() == 1:
        color.set('green')
    else:
        color.set('red')


def callback(event):
    if not cleared.get():
        changeOutputFile()
        cleared.set(True)
    x = event.x
    y = event.y
    # print("x:", x, " y:", y)
    x1, y1 = (x - 1), (y - 1)
    x2, y2 = (x + 1), (y + 1)
    w.create_oval(x1, y1, x2, y2, fill=color.get(), outline=color.get())
    dots.append([event.x/WIDTH,event.y/HEIGHT])
    groups.append([positive.get() if group.get() == 0 else negative.get(),positive.get() if group.get() == 1 else negative.get()])
    # with open(filename.get(), 'a+') as outfile:
    #     outfile.write(
    #         str(event.x / WIDTH) + ":" +
    #         str(event.y / HEIGHT) + "::::" +
    #         str(positive.get() if group.get() == 0 else negative.get()) + ":" +
    #         str(positive.get() if group.get() == 1 else negative.get()) + "\n")
    # print(str(event.x / WIDTH) + ":" +
    #     str(event.y / HEIGHT) + "::::" +
    #     str(positive.get() if group.get() == 0 else negative.get()) + ":" +
    #     str(positive.get() if group.get() == 1 else negative.get()) + "\n")


def changeOutputFile(event=None):
    filename.set(text.get())
    # open(text.get(), 'w')
    print(dots)
    print(groups)
    np.save(text.get()+"_in",np.array(dots,dtype=float))
    np.save(text.get()+"_out",np.array(groups,dtype=float))


w.focus_force()
w.bind("<Button-1>", callback)
w.bind("<Key>", toogleGroup)
button1 = Button(master, text="Toggle group", command=toogleGroup, fg='black')
button2 = Button(master, text="Exit", command=lambda event=None: exit(0))
text = Entry(master, textvariable=filename)
button3 = Button(master, text="Save", command=changeOutputFile)

negative_default = StringVar()
negative_default.set(-1)
positive_default = StringVar()
positive_default.set(1)

negative = Entry(master, textvariable=negative_default)
positive = Entry(master, textvariable=positive_default)

w.grid(row=0, column=0, columnspan=2)
button1.grid(row=1, column=0)
button2.grid(row=1, column=1)
text.grid(row=2, column=0)
button3.grid(row=2, column=1)
negative.grid(row=3, column=0)
positive.grid(row=3, column=1)
mainloop()
