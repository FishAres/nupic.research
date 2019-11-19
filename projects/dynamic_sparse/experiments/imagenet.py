# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------

# investigate how to load and save the dataset
# need to 
# 1 - loop once through the dataset to apply the transforms
# 2 - save each batch of images
# 3 - write a custom dataset that load from these saved files instead
# https://discuss.pytorch.org/t/save-transformed-resized-images-after-dataloader/56464/6

# file to test run imagenet

from torchvision import models, datasets, transforms
from torch import nn, utils
import os
from time import time

SMALL_IMAGENET = False

# local test 
# train_path = os.path.expanduser("~/nta/datasets/tiny-imagenet-200/train")
# val_path = os.path.expanduser("~/nta/datasets/tiny-imagenet-200/val")

train_path = os.path.expanduser("~/nta/data/imagenet/train")
val_path = os.path.expanduser("~/nta/data/imagenet/val")

if SMALL_IMAGENET:
    train_path = os.path.expanduser("~/nta/datasets/imagenet/train")
    val_path = os.path.expanduser("~/nta/datasets/imagenet/val")

# cifar10 stats
# stats_mean = (0.4914, 0.4822, 0.4465)
# stats_std = (0.2023, 0.1994, 0.2010)
# imagenet stats
stats_mean = (0.485, 0.456, 0.406)
stats_std = (0.229, 0.224, 0.225)
# preprocessing: https://github.com/pytorch/vision/issues/39
train_transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),    
    transforms.ToTensor(),
    transforms.Normalize(stats_mean, stats_std),
])

val_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(stats_mean, stats_std),
])

# dataset = datasets.CIFAR10(root=os.path.expanduser("~/nta/datasets"),
#     transform=transform, download=True)
# data_loader = utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# how to save and load data on the fly

# load train dataset
t0 = time()
train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
print("Loaded train dataset")
t1 = time()
print("Time spent to load train dataset: {:.2f}".format(t1-t0))

# load test dataset
t0 = time()
test_dataset = datasets.ImageFolder(val_path, transform=val_transform)
print("Loaded test dataset")
t1 = time()
print("Time spent to load test dataset: {:.2f}".format(t1-t0))

t0 = time()
train_dataloader = utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
print("Loaded train dataloader")
test_dataloader = utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)
print("Loaded test dataloader")
t1 = time()
print("Time spent to load dataloaders: {:.2f}".format(t1-t0))

# import pdb; pdb.set_trace()

t0 = time()
network = models.resnet50(pretrained=True)
# required to remove head if smaller dataset
if SMALL_IMAGENET:
    last_layer_shape = network.fc.weight.shape
    network.fc = nn.Linear(last_layer_shape[1], 5)
print("Loaded network")
t1 = time()
print("Time spent to load network: {:.2f}".format(t1-t0))

# ------------------------- RUN MODEL

# load the model
from nupic.research.frameworks.dynamic_sparse.common.datasets import CustomDataset
from nupic.research.frameworks.dynamic_sparse.models import BaseModel

# simple base model
t0 = time()
exp_config = dict(device='cuda')
model = BaseModel(network, exp_config)
model.setup()
# simple dataset
dataset  = CustomDataset(exp_config)
dataset.set_loaders(train_dataloader, test_dataloader)
epochs = 5
t1 = time()
print("Time spent to setup experiment: {:.2f}".format(t1-t0))
for epoch in range(epochs):
    t0 = time()
    print("Running epoch {}".format(str(epoch)))
    log = model.run_epoch(dataset, epoch)
    t1 = time()
    print("Train acc: {:.4f}, Val acc: {:.4f}".format(log['train_acc'], log['val_acc']))
    print("Time spent in epoch: {:.2f}".format(t1-t0))

# # ------------------------- RAY LOOP 
# from ray import tune
# import ray
# from nupic.research.frameworks.dynamic_sparse.common.experiments import CustomTrainable 
# from nupic.research.frameworks.dynamic_sparse.common.ray_custom_loggers import DEFAULT_LOGGERS

# exp_config['unpack_params'] = lambda: (model, dataset)
# ray.init()
# tune.run(CustomTrainable, 
#     name='imagenet-testscript',
#     config=exp_config, 
#     num_samples=1, 
#     resources_per_trial={"cpu": 1, "gpu": 1},
#     local_dir=os.path.expanduser("~/nta/results"),
#     checkpoint_freq=0,
#     checkpoint_at_end=False,
#     stop={"training_iteration": 3},    
#     loggers=DEFAULT_LOGGERS,
#     verbose=2,    
# )
# ray.shutdown()


# ------------------------- DEPRECATED (OUTER TRAINING LOOP)

# loss_func = nn.CrossEntropyLoss()
# print("Loaded model")



    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res