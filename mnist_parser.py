import numpy as np

def read_int(file, bytes_count):
    num=file.read(bytes_count)
    num_int =int.from_bytes(num, byteorder='big')
    return  num_int

def int_to_one_hot(num):
    one_hot = np.zeros(10)
    one_hot.fill(0)
    one_hot[num] = 1
    return one_hot

def one_hots_to_ints(one_hots):
    rez = np.array([])
    for one_hot in one_hots:
        num=one_hot_to_int(one_hot)
        rez=np.append(rez,num)
    return rez


def one_hot_to_int(one_hot):
    max=one_hot[0]
    num=0
    for i in range(1,10):
        if one_hot[i] > max:
            max=one_hot[i]
            num=i
    return num

def display_image(image,one_hot_number,min_color=-1,max_color=1):
    print("number:",one_hot_to_int(one_hot_number))
    interval = max_color-min_color
    quarter = interval/4

    for i in range(0,28):
        for j in range(0,28):
            if image[i*28+j] > min_color+3*quarter:
                print("O",end='')
            elif image[i*28+j] > min_color+2*quarter:
                print("o",end='')
            elif image[i*28+j] > min_color+quarter:
                print(".",end='')
            else:
                print(" ",end='')

        print("")

def load_mnist_database(images_file,labels_file,max=None):
    #load_labels
    labels_file= open(labels_file,"rb")
    try:
        magic_num_lab=read_int(labels_file, 4)
        print("Magic number:",magic_num_lab)

        items_count=read_int(labels_file,4)
        print("Number of labels:", items_count)

        if max is not None:
            items_count=max

        labels_matrix = np.zeros(shape=(1, 10))
        byte = labels_file.read(1)
        for i in range(0,items_count):
            print("labels loaded:",i, "/",items_count)

            number = int.from_bytes(byte,byteorder='big')

            label_one_hot=int_to_one_hot(number)

            labels_matrix=np.concatenate([labels_matrix,[label_one_hot]])


            byte=labels_file.read(1)

    finally:
        labels_file.close()


    #load images
    images_file = open(images_file, 'rb')
    try:

        magic_num_img=read_int(images_file, 4)
        print("Magic number:",magic_num_img)

        items_count=read_int(images_file, 4)
        print("Number of items:", items_count)

        if max is not None:
            items_count=max

        rows=read_int(images_file, 4)
        print("Number of rows:", rows)

        columns=read_int(images_file, 4)
        print("Number of columns:", columns)

        flat_image_size=rows*columns
        images_matrix = np.zeros(shape=(1,flat_image_size))

        byte=images_file.read(1)

        index=0
        image=np.zeros(flat_image_size)

        for i in range(0,(items_count+1)*flat_image_size):
            if(i % (10*flat_image_size) == 0):
                print("Images loaded",i//flat_image_size,"/",items_count)
            if(index==flat_image_size):
                #save previous image
                # display_image(image,labels_matrix[i//flat_image_size])
                images_matrix = np.concatenate([images_matrix, [image]])

                # next image
                image = np.zeros(flat_image_size)
                index=0

            pixel = int.from_bytes(byte, byteorder='big')
            pixel = (pixel/127.5) - 1 # pretvaramo interval 0 - 255 na -1 - 1
            image[index]=pixel

            byte=images_file.read(1)
            index+=1
    finally:
        images_file.close()





    images_matrix = np.delete(images_matrix, [0], axis=0)
    labels_matrix = np.delete(labels_matrix, [0], axis=0)
    return images_matrix, labels_matrix


def parse_to_npy(images_file,labels_file,numpy_images="mnist_images",numpy_labels="mnist_labels"):
    images,labels = load_mnist_database(images_file,labels_file)
    np.save(numpy_images,images)
    np.save(numpy_labels,labels)
