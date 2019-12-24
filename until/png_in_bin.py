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
    

