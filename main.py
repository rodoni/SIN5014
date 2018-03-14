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

img = mpimg.imread('/home/rodoni/cat.jpeg')
print(img.shape)
gray = rgb2gray(img)
print(gray)
print(gray.item(0, 0))

[B, G, R] = cv2.split(img)

#img_concat = np.append([B], [G], [R], axis=0)
#print(img_concat.shape)
merged_image = cv2.merge((B, G, R))
#show_separeted_channels(B, G ,R )

plt.imshow(merged_image)
plt.show()

print(gray.shape)
print(B.shape)


plt.show()

