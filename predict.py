import torch

import os
import argparse

from Data import Data

# setup arguments
parser = argparse.ArgumentParser(description="Classify input image.")
parser.add_argument("-m", "--model", type=str, required=True,
                    help="The path of the trained and validated model. For example '~/opt/VGG16_trained.pth'")
parser.add_argument("-i", "--impath", type=str,
                    help="The path of the image to be classified. For example '/test/1/image_06743.jpg'")
parser.add_argument("-k", "--top_k", type=int,
                    help="The number of top-k probabilities to return from the classification. For example 5")
parser.add_argument("-g", "--gpu", type=bool,
                    help="Whether to attempt to use the GPU for classification or not. NOTE: this flag is deppricated and doesn't do anything, if a GPU is available, it will be used automatically.")

# parse arguments
args = parser.parse_args()

# manager directory
cwd = os.getcwd()
root_dir = cwd + "/flowers"
data = Data(root_dir)

if args.impath:
    if root_dir in args.impath:
        image_path = args.impath
    else:
        image_path = root_dir + args.impath
    image, idx = data.get_image_from_path(image_path)
else:
    image, idx = data.get_random_test_image()

print(idx, data.get_name(index=idx))