# A Tensor is a container for data — almost always numerical data.
# E.g. Matrices are 2D tensors

# Scalars - 0D tensors
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
x = np.array(12)
print(x)
    # ndim displays the dimensions/rank of the tensor
print(x.ndim)

# Vectors - 1D tensors
x = np.array([12, 3, 6, 14, 7])
print(x)
print(x.ndim)
    # Note that this is a 5D vector, not a 5D tensor.

# Matrices - 2D tensors
x = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]])
print(x.ndim)

# 3D tensors and higher-dimensional tensors
x = np.array([[[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]]])
print(x.ndim)

# A Tensor has 3 key attributes:
    # - Number of axes (rank)
    # - Shape
    # - Data type

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# display the number of axes of the tensor train_images, the ndim attribute
print(train_images.ndim)
# display the shape of the tensor train_images, the shape attribute
print(train_images.shape)
# display the data type of the tensor train_images, the dtype attribute
print(train_images.dtype)

# Displaying the fourth digit in a 3D tensor
digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
    # train_images is an array containing multiple images. Each image in this array is itself a 2D array of pixel values.
    # The variable digit now holds the data for this fifth image.
    # Matplotlib is a library for creating visualizations, including plotting images.
    # plt.imshow is a function that displays data as an image. Here, it's used to display the contents of digit.
    # digit is the 2D array of pixel values representing the image.
    # cmap=plt.cm.binary specifies the color map to use.
    # The 'binary' colormap displays the image in grayscale, interpreting the array values as intensities (0 being black, 255 being white for 8-bit grayscale images).

# Manipulating tensors in Numpy
    # In the previous example, we chose a specific digit alongside the first axis using the syntax train_images[i].
    # This is called tensor slicing
    # E.g: Selects digits #10 to #100 (#100 not included) and puts them in an array of shape (90, 28, 28):
my_slice = train_images[10:100]
print(my_slice.shape)
    # When you slice a portion of a multidimentional array, it retains the original dimensions of the array.
    # It's equivalent to this more detailed notation, which specifies a start index and stop index for the slice along each tensor axis.
my_slice = train_images[10:100, :, :]
my_slice = train_images[10:100, 0:28, 0:28]
    # Deep learning model don;t process an entire dataset at once; they break them into small batches;
        # batch = train_images[:128]
    # Next batch
        # batch = train_images[128:256]



