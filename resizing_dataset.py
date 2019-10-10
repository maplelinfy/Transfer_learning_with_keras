##USAGE

#python resizing_dataset.py --dataset path_to _dataset --storage locaton_for_output

from imutils import paths
import numpy as np
import argparse
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-s", "--stotage", required=True,
    help="path for output of the program (i.e., directory of output images)")
IMAGE_DIMS = (150, 150, 3)

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	try:
		image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
		image = img_to_array(image)
        cv2.imwrite(args["storage"],image)
	except:
		os.remove(imagePath)

print("[INFO] done with image resizing...")
