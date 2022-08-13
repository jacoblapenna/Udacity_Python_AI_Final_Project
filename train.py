import torch
from torchvision import models

import os
import time
import argparse
from pathlib import Path

from Data import Data
from Trainer import Trainer
from Validator import Validator
from workspace_utils import active_session

# setup arguments
parser = argparse.ArgumentParser(description="Train and validate a CNN classifier.")
parser.add_argument("data_dir", type=str,
                    help="The path to the image folder. For example '/flowers'")
parser.add_argument("-s", "--save_dir", type=str,
                    help="The directory and filename to save the trained model. For example: '~/opt/models/model_name.pth'")
parser.add_argument("-a", "--arch", type=str,
                    help="The architecture to train. The following pytorch architectures are available: 'alexnet', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50'")
parser.add_argument("-lr", "--learn_rate", type=float,
                    help="The learning rate for training the network. For example: 0.003")
parser.add_argument("-m", "--momentum", type=float,
                    help="The momentum to use with the SGD optimizer. For example: 0.9")
parser.add_argument("-e", "--epochs", type=int,
                    help="The number of epochs to use. For example: 30")
parser.add_argument("-c", "--checkpoint_step", type=int,
                    help="The number of epochs at which checkpoints should be saved intermittently during training. For example: 3")
parser.add_argument("-g", "--gpu", type=bool,
                    help="Whether to attempt to use the GPU for classification or not. NOTE: this flag is deppricated and doesn't do anything, if a GPU is available, it will be used automatically.")

# parse arguments
args = parser.parse_args()

# manage paths
cwd = os.getcwd()
root_dir = cwd
# data path
if root_dir in args.data_dir:
    data_path = args.data_dir
else:
    data_path = root_dir + args.data_dir
# checkpoint path
if os.path.isdir("/root/opt/models"):
    pass
else:
    os.mkdir("~/opt/models")
if args.save_dir:
    if "/root/opt/models" in args.save_dir:
        save_path = args.save_dir
    else:
        save_path = "/root/opt/models" + args.save_dir
try:
    Path('/'.join(save_path.split('/')[:-1])).touch(save_path.split('/')[-1])
except FileNotFoundError:
    raise Exception(f"Bad save path {save_path}")
    
# get data
data = Data(data_path)

# get model
if args.arch not in ['alexnet', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50']:
    raise Exception("Invalid architecture! Try -h for help.")
model = eval(f"models.{args.arch}(pretrained=True)")

# freeze features
for param in model.parameters():
    # turn off tracking for gradient to freeze parameters
    param.requires_grad = False

# modify classifier to match number of possible classesifications
model.classifier[-1].out_features = data.n_outputs
# make sure classifier is not frozen
for param in model.classifier.parameters():
    param.requires_grad = True

# set the device to detect hardware and prefer a GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# send model to device
model.to(device)

# setuo hyper params
if args.epochs:
    epochs = args.epochs
else:
    epochs = 30
if args.learn_rate:
    learnrate = args.learn_rate
else:
    learnrate = 0.001
if args.momentum:
    momentum = args.momentum
else:
    momentum = 0.9
checkpoint = args.checkpoint_step  

# get trainer instance
trainer = Trainer(data.trainloader, model, device, learnrate, momentum)
# get validator instance
validator = Validator(data.valloader, device)

# train and validate
with active_session():
    ## perform training
    # run through each epoch
    for epoch in range(epochs):
        print(f"{time.strftime('%H:%M:%S', time.localtime())} - Training network on epoch {epoch}...")
        trainer.train()
        print(f"{time.strftime('%H:%M:%S', time.localtime())} - Validating network on epoch {epoch}...")
        validator.validate(trainer.network, trainer.criterion)
    
        # show user epoch results
        tl = trainer.training_losses[-1]
        vl = validator.validation_losses[-1]
        a = validator.accuracies[-1]
        print(f"{time.strftime('%H:%M:%S', time.localtime())} - Epoch {epoch} results: training_loss={tl}, validation_loss={vl}, accuracy={a}")

        # save every checkpoint step
        if checkpoint:
            if epoch % checkpoint == 0:
                torch.save(trainer.network, ('/').join(save_path.split('/')[:-1]) + f"/{args.model}_at_epoch_{epoch}.pth")
    

torch.save(trainer.network, save_path)
