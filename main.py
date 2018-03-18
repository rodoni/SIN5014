import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import cv2


# to split RGB channels, this returns B, G, R
def channel_split(image_to_split):

    [B, G, R] = np.dsplit(image_to_split, image_to_split.shape[-1])
    B = np.squeeze(B, axis=2)
    G = np.squeeze(G, axis=2)
    R = np.squeeze(R, axis=2)
    return [B, G, R]

# to rgb to gray
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

# show separeted images
def show_separeted_channels(B, G, R):

    plt.imshow(B, cmap=plt.get_cmap('Blues'))
    plt.show()
    plt.imshow(G, cmap=plt.get_cmap('Greens'))
    plt.show()
    plt.imshow(R, cmap=plt.get_cmap('Reds'))
    plt.show()


# histogram
def histogram(channel, resolution ):

    matrix = np.array(channel)
    matrix = matrix.astype(int)
    lines, columns = matrix.shape
    histogram_array = np.zeros(resolution)

    for line in range(lines):
        for column in range(columns):
            histogram_array[matrix.item(line, column)] = histogram_array[matrix.item(line, column)] + 1

    return histogram_array


# show histogram_graph
def plot_histogram(histogram_array):

    x_axis = np.arange(histogram_array.size)
    plt.bar(x_axis, histogram_array,0.5)
    plt.show()


# change the intensity of one channel
def change_intesity(matrix, parameter):

    lines, columns = matrix.shape

    for line in range(lines):
        for column in range(columns):
            temp = matrix.item(line, column) + parameter

            if temp > 255:
                matrix[line][column] = 255
            elif temp < 0:
                matrix[line][column] = 0
            else:
                matrix[line][column] = temp

    return matrix


def smoothing_average( matrix ):

    lines, columns = matrix.shape

    for line in range(lines):
        for column in range(columns):
            mask = []

            for x in -1, 0, 1:
                for y in -1, 0, 1:
                    if line + x >= 0 and line +x < lines and column + y >= 0 and column + y < columns:
                        mask.append(matrix.item(line+x, column+y))

            matrix[line][column] = sum(mask)/len(mask)

    return matrix


def smoothing_median( matrix ):

    lines, columns = matrix.shape
    for line in range(lines):
        for column in range(columns):
            mask = []

            for x in -1, 0, 1:
                for y in -1, 0, 1:
                    if line + x >= 0 and line +x < lines and column + y >= 0 and column + y < columns:
                        mask.append(matrix.item(line+x, column+y))

            mask = np.sort(mask)
            index = len(mask)/2
            matrix[line][column] = mask[int(index)]

    return matrix


def equalization ( matrix, histogram_image ):

    size = histogram_image.size
    histogram_cumulative = np.zeros(size)
    equalized_grey_level = np.zeros(size)
    histogram_cumulative[0] = histogram_image[0]
    lines, columns = matrix.shape
    p = (lines*columns)/255

    # cumulative histogram
    for grey_level in range(1, histogram_image.size - 1):
        histogram_cumulative[grey_level] = histogram_cumulative[grey_level-1] + histogram_image[grey_level]

    for grey_level in range(histogram_cumulative.size):
        columative = histogram_cumulative[grey_level]
        if columative == 0:
            equalized_grey_level[grey_level] = 0
        else:
            equalized_grey_level[grey_level] = max(0, int(histogram_cumulative[grey_level]/p - 1))

    for line in range(lines):
        for column in range(columns):
            grey_level = int(matrix.item(line, column))
            matrix[line][column] = equalized_grey_level[grey_level]

    return matrix



# Check input variable
if len(sys.argv) > 1:
    image_path = sys.argv[1]
    if len(sys.argv) > 2:
        options = sys.argv[2]
        if len(sys.argv) > 3:
            parameters = sys.argv[3]
else:
    exit()


img = mpimg.imread(image_path)
gray = rgb2gray(img)
plt.imshow(gray, cmap= "Greys_r")
plt.show()

if options == "-histogram":

    histogram_gray = histogram(gray, 256)
    plot_histogram(histogram_gray)

if options == "-intensity":

    img_modifed = change_intesity(gray, int(parameters))
    plt.imshow(img_modifed, cmap="Greys_r")
    plt.show()

if options == "-average_filter":

    img_modifed  = smoothing_average(gray)
    plt.imshow(img_modifed, cmap="Greys_r")
    plt.show()

if options == "-median_filter":

    img_modifed  = smoothing_median(gray)
    plt.imshow(img_modifed, cmap="Greys_r")
    plt.show()

if options == "-equalization":

    histogram_gray = histogram(gray, 256)
    img_modifed  = equalization(gray, histogram_gray)
    plt.imshow(img_modifed, cmap="Greys_r")
    plt.show()


exit()




