import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np

# Set paths
voc_dir = "/home/sylvex/Downloads/pascal/VOC2012_train_val/VOC2012_train_val"  # Update to your dataset path
image_dir = os.path.join(voc_dir, "JPEGImages")
annotation_dir = os.path.join(voc_dir, "Annotations")
train_file = os.path.join(voc_dir, "ImageSets/Main/val.txt")

# Load the list of training images
with open(train_file, "r") as f:
    image_ids = f.read().strip().split()

# Function to parse XML annotation
def parse_voc_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get image filename
    filename = root.find("filename").text
    image_path = os.path.join(image_dir, filename)

    # Load image (optional, for visualization or processing)
    image = Image.open(image_path)

    return image

# # Example: Load and parse one image
# sample_image_id = image_ids[69]
# xml_path = os.path.join(annotation_dir, f"{sample_image_id}.xml")
# image_path, image, objects = parse_voc_xml(xml_path)

# print(f"Image path: {image_path}")
# print(f"Objects: {objects}")

# Visualize the image with bounding boxes
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_letterbox_label(label, original_sizes, target_size):
    # Get the original image size
    original_w, original_h = original_sizes

    # Get the target size
    target_w, target_h = target_size

    # Compute the scale factor
    scale = min(target_w / original_w, target_h / original_h)

    # Compute the padding
    pad_w = (target_w - original_w * scale) / 2
    pad_h = (target_h - original_h * scale) / 2

    # Scale the bounding box coordinates
    label["xmin"] = (label["xmin"] * scale + pad_w)
    label["ymin"] = (label["ymin"] * scale + pad_h)
    label["xmax"] = (label["xmax"] * scale + pad_w)
    label["ymax"] = (label["ymax"] * scale + pad_h)

    return label

def save_image(image, output_path="output.png"):
    # image is normalized, so we need to denormalize it
    image = np.transpose(image[0], [1, 2, 0])
    image = (image).astype(np.uint8)
    image = Image.fromarray(image) # 416x416

    image.save(output_path)
    

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def preprocess(img):
    img = img.resize((224, 224), Image.BICUBIC)
    image_data = np.array(img, dtype='float32')
    # resize to 224x224
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data

def get_random_sample(num_samples):
    images = []
    for i in range(num_samples):
        image_id = np.random.choice(image_ids)
        xml_path = os.path.join(annotation_dir, f"{image_id}.xml")
        image = parse_voc_xml(xml_path)
        image = preprocess(image)
        images.append(image)
    return images

if __name__ == "__main__":
    np.random.seed(42069)

    images = get_random_sample(2)
    print(f"image shapes: {[img.shape for img in images]}")
    save_image(images[0], "image0.png")
    save_image(images[1], "image1.png")