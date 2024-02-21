import os
import sys, getopt
from tqdm import tqdm
import random
from skimage.transform import resize
from skimage.io import imread, imsave
import numpy as np
import shutil

def processArguments(argv: list) -> tuple[str, str, float, int]:
    params = {
        "input_folder": os.getcwd(),
        "output_folder": os.getcwd(),
        "train_size": 10000,
        "test_size": 1000,
        "validation_size": 500,
        "img_size": 256,
    }
    try:
        opts, args = getopt.getopt(argv,"hi:o:r:s:",["ifolder=","ofolder=", "train_size=", "test_size=", "validation_size=",  "img_size="])
    except getopt.GetoptError:
        print(getopt.GetoptError)
        print("'processImages.py -i <inputfile> -o <outputfile> -s <seed> -r <train_test_ratio>'")
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-i", "--ifolder"):
            params["input_folder"] = arg
        if opt in ("-o", "--ofolder"):
            params["output_folder"] = arg
        if opt in ("-ts", "--train_size"):
            params["train_size"] = int(arg)
        if opt in ("-te", "--test_size"):
            params["test_size"] = int(arg)
        if opt in ("-vl", "--cle"):
            params["validation_size"] = int(arg)
        if opt in ("-is", "--img_size"):
            params["img_size"] = int(arg)
    print(params)
    return params

def transform_and_save(images: set, n:int, output_folder: str, source_folder:str, img_size:int = 256) -> set[str]:
    i = 0
    selected = []
    images = list(images)
    while i < n:
        image_source = f"{source_folder}/{images[i]}"
        image_destination = f"{output_folder}/{images[i]}"
        image = imread(image_source)
        if len(image.shape) == 3:
            image = resize(image, (img_size,img_size), anti_aliasing=True)
            image = (image*255).astype(np.uint8)
            imsave(image_destination, image)
            selected.append(images[i])
            i += 1
        else:
            images.remove(images[i])
    return set(selected)
        

def main(argv: list):
    params = processArguments(argv)
    output_folder = params["output_folder"]
    train_dir = f"{output_folder}/train"
    test_dir = f"{output_folder}/test"
    validation_dir = f"{output_folder}/validation"
    #Prepare folder structure
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
        os.makedirs(train_dir)
        os.makedirs(test_dir)
    else:
        if os.path.isdir(train_dir):
            shutil.rmtree(train_dir)
        if os.path.isdir(test_dir):
            shutil.rmtree(test_dir)
        if os.path.isdir(validation_dir):
            shutil.rmtree(validation_dir)
        os.makedirs(train_dir)
        os.makedirs(test_dir)
        os.makedirs(validation_dir)


    #train test split
    images = os.listdir(params["input_folder"])
    random.seed(12325298)
    random.shuffle(images)
    images = set(images)

    train_images = transform_and_save(images, params["train_size"], train_dir, params["input_folder"], params["img_size"], )
    images = images.difference(train_images)
    test_images = transform_and_save(images,  params["test_size"], test_dir, params["input_folder"], params["img_size"]) 
    images = images.difference(test_images)
    transform_and_save(images,  params["validation_size"], validation_dir, params["input_folder"], params["img_size"])
    shutil.make_archive(params["output_folder"], 'zip', params["output_folder"])
        
        
if __name__ == "__main__":
    main(sys.argv[1:])
