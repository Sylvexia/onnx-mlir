import numpy as np
import os
from PIL import Image

def preprocess(image):
    # resize so that the shorter side is 256, maintaining aspect ratio
    def image_resize(image, min_len):
        image = Image.fromarray(image)
        ratio = float(min_len) / min(image.size[0], image.size[1])
        if image.size[0] > image.size[1]:
            new_size = (int(round(ratio * image.size[0])), min_len)
        else:
            new_size = (min_len, int(round(ratio * image.size[1])))
        image = image.resize(new_size, Image.BILINEAR)
        return np.array(image)
    image = image_resize(image, 256)

    # Crop centered window 224x224
    def crop_center(image, crop_w, crop_h):
        h, w, c = image.shape
        start_x = w//2 - crop_w//2
        start_y = h//2 - crop_h//2
        return image[start_y:start_y+crop_h, start_x:start_x+crop_w, :]
    image = crop_center(image, 224, 224)

    # transpose
    image = image.transpose(2, 0, 1)

    # convert the input data into the float32 input
    img_data = image.astype('float32')

    # normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]

    # add batch channel
    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    return norm_img_data

# get a random numpy array and label number from imagenet dataset path
def get_random_imagenet(dataset_path, num_samples=1):
    # directory name as label
    labels = sorted(os.listdir(dataset_path))
    label_index = {label: i for i, label in enumerate(labels)}

    # get all image filename and populate filename -> label mapping
    label_map = {}
    for label in labels:
        image_files = os.listdir(os.path.join(dataset_path, label))
        for image_file in image_files:
            label_map[image_file] = label

    # get random image
    image_files = np.random.choice(list(label_map.keys()), num_samples)
    return_label_names = [label_map[image_file] for image_file in image_files]
    return_label_indices = [label_index[label_name] for label_name in return_label_names]

    # for each image, read the image and preprocess
    images = []
    for image_file in image_files:
        image = Image.open(os.path.join(dataset_path, label_map[image_file], image_file))
        image = np.array(image)
        images.append(preprocess(image))

    return images, return_label_indices

DATAPATH = '/home/sylvex/Downloads/imagenet-mini/val'

if __name__ == '__main__':
    np.random.seed(42069)
    images, labels = get_random_imagenet(DATAPATH, 2)
    print(images)
    print(labels)
    # iterate over images and save them
    for i, image in enumerate(images):
        image = image.reshape(3, 224, 224).transpose(1, 2, 0)
        image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        image.save(f'imagenet_{labels[i]}.png')
