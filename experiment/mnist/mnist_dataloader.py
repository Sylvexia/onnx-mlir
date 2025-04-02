import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
import matplotlib.pyplot as plt

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, test_images_filepath, test_labels_filepath):
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        # shuffle test data
        p = np.random.permutation(len(x_test))
        x_test = np.array(x_test)[p]
        y_test = np.array(y_test)[p]
        return (x_test, y_test)

#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        # plt.imshow(image, cmap=plt.cm.gray)
        # save image
        plt.imsave(f'image{index}-{title_text}.png', image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1

def get_random_mnist(num_images = 5):
    #
    # Set file paths based on added MNIST Datasets
    #
    input_path = '/home/sylvex/Downloads/mnist_data/'
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    #
    # Load MINST dataset
    #
    mnist_dataloader = MnistDataloader(test_images_filepath, test_labels_filepath)
    (x_test, y_test) = mnist_dataloader.load_data()

    # resize to float32 and normalize
    x_test = np.array(x_test).astype('float32')/255
    x_test = x_test.reshape(x_test.shape[0], 1, 1, 28, 28)
    return x_test[:num_images], y_test[:num_images]                                              

#
# Show some random training and test images 
#

x_test, y_test = get_random_mnist(6)

# images_2_show = []
# titles_2_show = []

# for i in range(0, 5):
#     r = random.randint(1, 10000)
#     images_2_show.append(x_test[r])        
#     titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

# show_images(x_test, y_test)