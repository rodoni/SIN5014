import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
    histogram_array = np.zeros(resolution).astype(int)

    for line in range(lines):
        for column in range(columns):
            histogram_array[matrix.item(line, column)] = histogram_array[matrix.item(line, column)] + 1

    return histogram_array


# show histogram_graph
def plot_histogram(histogram_array):

    x_axis = np.arange(histogram_array.size)
    plt.bar(x_axis, histogram_array,0.5)
    plt.show()


# change the intensity
def change_intesity(matrix, parameter):

    # if the image is rgb
    if matrix.ndim > 2:
        lines, columns, channels = matrix.shape

        for line in range(lines):
            for column in range(columns):
                for channel in range(channels):
                    temp = matrix.item(line, column, channel) + parameter
                    if temp > 255:
                        matrix[line][column][channel] = 255

                    elif temp < 0:
                        matrix[line][column][channel] = 0

                    else:
                        matrix[line][column][channel] = temp

    # if the image is in gray scale
    else:
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



img = mpimg.imread('/home/rodoni/cat.jpeg')
#print(img.shape)
gray = rgb2gray(img)



img_modified = np.array(img)
img_modified = change_intesity(img_modified, -50)
print(img_modified[0][0][0])
print(img[0][0][0])

plt.imshow(img_modified)
plt.show()

plt.imshow(img)
plt.show()



#print(gray.item(0, 0))

[B, G, R] = cv2.split(img)

#print(img[0][1][1])
#print(img.item(0,1,2))
#print(img)

#img_concat = np.append([B], [G], [R], axis=0)
#print(img_concat.shape)
merged_image = cv2.merge((B, G, R))
#print(gray.shape)

#histogram_array = histogram(gray, 256)
#print(histogram_array.shape)
#plot_histogram(histogram_array)

#print(histogram_array.shape)
#print(histogram_array)

#plt.imshow(R,cmap = "Reds")
#plt.show()

#print(gray.shape)
#print(B.shape)


#plt.show()

