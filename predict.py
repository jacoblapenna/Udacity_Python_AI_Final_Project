import torch

import os
import argparse

from Data import Data

"""
NOTES:
Can't download pth files from the server of upload to github. Found AlexNet accuracy at 74.75% with 30 epochs, learnrate of 0.001 and SGD momentum of 0.9.

"""

# setup arguments
parser = argparse.ArgumentParser(description="Classify input image.")
parser.add_argument("-m", "--model", type=str, required=True,
                    help="The path of the trained and validated model. For example '~/opt/VGG16_trained.pth'")
parser.add_argument("-i", "--impath", type=str,
                    help="The path of the image to be classified. For example '/test/1/image_06743.jpg'")
parser.add_argument("-k", "--top_k", type=int,
                    help="The number of top-k probabilities to return from the classification. For example 5")
parser.add_argument("-c", "--category_names", type=str,
                    help="Allows for alternative mapping of classes to names. For example 'class_to_names.json'")
parser.add_argument("-g", "--gpu", type=bool,
                    help="Whether to attempt to use the GPU for classification or not. NOTE: this flag is deppricated and doesn't do anything, if a GPU is available, it will be used automatically.")

# parse arguments
args = parser.parse_args()

# manage paths
cwd = os.getcwd()
root_dir = cwd + "/flowers"

# create Data instance
data = Data(root_dir)

# get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# soft max
output = torch.nn.LogSoftmax(dim=1)

# get image and classifier
if args.impath:
    if root_dir in args.impath:
        image_path = args.impath
    else:
        image_path = root_dir + args.impath
    image, idx = data.get_image_from_path(image_path)
else:
    image, idx = data.get_random_test_image()
image = image.to(device)

# get label
label = data.get_name(index=idx)

# load model on best available device and set to eval mode
# map device location
if torch.cuda.is_available():
    map_location = lambda memory, location: memory.cuda()
else:
    map_location = "cpu"
# load model
model = torch.load(args.model, map_location=map_location)


# classify
with torch.no_grad():
    model.eval() # prevent dropout
    log_ps = output(model.forward(image))
    ps = torch.exp(log_ps)

# get topk
if args.top_k:
    top_probs, top_classes = ps.topk(args.top_k, dim=1)
else:
    top_probs, top_classes = ps.topk(1, dim=1)
# process topk
top_probs, top_classes = top_probs.tolist()[0], top_classes.tolist()[0]
# get top names from top classes
if args.category_names:
    data.set_label_to_names(args.category_names)
top_names = [data.get_name(index=cls) for cls in top_classes]

# inform user of results
roundl = lambda x: round(x, 3)
if args.top_k and args.top_k > 1:
    print(f"The top {args.top_k} probabilities are: {list(map(roundl, top_probs))}, which correspond to the following flowers: {top_names}")
else:
    print(f"The flower was classified as {top_names[0]} with {roundl(top_probs[0])} probability.")
success_string = f"The actual flower was labeled {label}, "
if label == top_names[0]:
    success_string += f"which matches the highest probability classification of {top_names[0]}."
else:
    success_string += f"which does NOT match the highest probability classification of {top_names[0]}."
print(success_string)
