# # import click
# from test_new.test import qwe

# #print(qwe())
# def tyu():
import imageio


def png_in_str():
    img_data = imageio.imread('py_png.png')
    lst_data = list()

    for i in range(len(img_data)):
        for j in range(len(img_data)):
            if sum(img_data[i][j].tolist()) > 1000:
                lst_data.append('0')
            else:
                lst_data.append('1')  
    fin_str = lst_data
    fin_str = ''.join(lst_data)
    return (fin_str, lst_data)


def show_png_terminal():
    lst_data = png_in_str()[1]
    n = 0
    for i in range(len(lst_data)):
        n += 1
        if n == 28:
            lst_data[i] = lst_data[i] + '\n'
            n = 0
 
    str_data = ''.join(lst_data)
    print(str_data)
    

#print(type(png_in_str()[0]))

import numpy
import matplotlib
import matplotlib.pyplot as plt
image_data = png_in_str()[1]
image_data = [int(i) for i in image_data]
image_data = numpy.array(image_data)

print(image_data)

matplotlib.pyplot.imshow(image_data.reshape(28, 28), cmap='Greys', interpolation='None')
