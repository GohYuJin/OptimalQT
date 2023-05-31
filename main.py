# This set of code was copied over from the ipython notbook. Edits will be done there, this is mainly for easy viewing.

# Standard libraries
import itertools
import numpy as np
import matplotlib.pyplot as plt
# PyTorch
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import json
from PIL import Image
from copy import deepcopy
from torch.autograd import Variable
from tqdm import tqdm

from DiffJPEG import DiffJPEG
from layers import gumbel_softmax
from helpers import create_default_qtables, return_class_name, return_class_accuracy, visualize

resnet = torchvision.models.resnet50(pretrained=False) 
#Just use pretrained = True if you can download the weights
resnet.load_state_dict(torch.load('../weights/resnet50.pth')) 
resnet.eval()
for param in resnet.parameters():
    param.requires_grad = False
print()


f = open("class_lists/imagenet_class_index.json")
id_classname_json = json.load(f)
preprocess = transforms.Compose([
                            # transforms.CenterCrop((crop_size[1],crop_size[0])),
                            transforms.Resize((256,256)),
                            transforms.ToTensor(),
                            ])
image = preprocess(Image.open("sample_imgs/panda.jpg"))
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
norm = transforms.Normalize(mean=mean, std=std)

y_table, c_table = create_default_qtables()
JPEGCompress = DiffJPEG(image.shape[1], image.shape[2], differentiable=True, quality=15)
compressed_image = JPEGCompress(image.unsqueeze(0), y_table, c_table)[0]
plt.imshow(transforms.ToPILImage()(compressed_image))

with torch.no_grad():
    predictions = resnet(norm(image).unsqueeze(0))
    (target_class, target_dim) = return_class_name(id_classname_json, predictions)
    print(target_class)
    acc_of_original = return_class_accuracy(predictions, target_dim)
    
    predictions = resnet(norm(compressed_image).unsqueeze(0))
    acc_of_compressed = return_class_accuracy(predictions, target_dim)
    print(acc_of_original, acc_of_compressed)
    # predictied_class = return_class_name(predictions)[0]
    
    
def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

DatasetWIdx = dataset_with_indices(torchvision.datasets.ImageFolder)
train_dataset = DatasetWIdx("../../data/imagenette2/train", transform = preprocess)
train_dataset.class_to_idx = {
 'n01440764': 0,
 'n02102040': 217,
 'n02979186': 482,
 'n03000684': 491,
 'n03028079': 497,
 'n03394916': 566,
 'n03417042': 569,
 'n03425413': 571,
 'n03445777': 574,
 'n03888257': 701}
train_dataset.samples = train_dataset.make_dataset(train_dataset.root, 
                                                   train_dataset.class_to_idx, 
                                                   train_dataset.extensions, 
                                                   None)

val_dataset = torchvision.datasets.ImageFolder("../../data/imagenette2/train", transform = preprocess)
val_dataset.class_to_idx = {
 'n01440764': 0,
 'n02102040': 217,
 'n02979186': 482,
 'n03000684': 491,
 'n03028079': 497,
 'n03394916': 566,
 'n03417042': 569,
 'n03425413': 571,
 'n03445777': 574,
 'n03888257': 701}
val_dataset.samples = val_dataset.make_dataset(val_dataset.root, 
                                               val_dataset.class_to_idx, 
                                               val_dataset.extensions, 
                                               None)

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle = True, batch_size = 256)
val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=16)


y_table, c_table = create_default_qtables()
# y_table = torch.ones((8,8))
# c_table = torch.ones((8,8))
train_acc = 0
for (image, labels, idx) in tqdm(train_loader):
    with torch.no_grad():
        compressed_image = JPEGCompress(image, y_table, c_table)
        logits = resnet(norm(compressed_image))
        (target_class, target_dim) = return_class_name(id_classname_json, logits[-1].unsqueeze(0))
        acc = return_class_accuracy(logits[-1].unsqueeze(0), target_dim)
        train_acc = train_acc + (logits.argmax(dim=1) == labels).sum()

print("Accuracy with default Q-Table:", train_acc/len(train_dataset))


max_q_value = 128

y_table, c_table = create_default_qtables()
# y_table_1hot = torch.nn.functional.one_hot(y_table.type(torch.LongTensor) - 1, num_classes=255).type(torch.FloatTensor)
# c_table_1hot = torch.nn.functional.one_hot(c_table.type(torch.LongTensor) - 1, num_classes=255).type(torch.FloatTensor)
y_table_1hot = torch.nn.functional.one_hot(torch.ones((8,8), dtype=torch.LongTensor.dtype), 
                                           num_classes=max_q_value-1).type(torch.FloatTensor)
c_table_1hot = torch.nn.functional.one_hot(torch.ones((8,8), dtype=torch.LongTensor.dtype), 
                                           num_classes=max_q_value-1).type(torch.FloatTensor) 

y_table_1hot.requires_grad = True
c_table_1hot.requires_grad = True

optimizer = torch.optim.NAdam([
    y_table_1hot, 
    c_table_1hot
], lr=0.1)

ones_table = torch.ones((8,8))
ones_table.requires_grad = False

categorical_values_table = torch.arange(max_q_value-1).reshape(1,-1) + 1 
# + 1 to avoid divide by zero errors when quantizing
categorical_values_table.requires_grad = False

# print(y_table.requires_grad)

loss = torch.nn.MSELoss()
qloss = torch.nn.MSELoss(reduction='mean')

# sanity check to ensure that gradients are flowing and changing the 1 hot y_table 
best_y_1hot = deepcopy(y_table_1hot) 

best_y_table = deepcopy(y_table)
best_c_table = deepcopy(c_table)
best_loss = np.inf

ori_train_acc = 0.0

initial_temperature = 0.5
temperature_anneal_rate = 0.05

# Hyperparam alpha to balance between maximizing Quantization and 
# minimizing euclidean dist against uncompressed logits
alpha = torch.tensor((1e-5)) 
alpha.requires_grad=True

logits_uncompressed = torch.zeros((len(train_dataset), resnet.fc.out_features))
for (image, labels, idx) in tqdm(train_loader):
    with torch.no_grad():
        ori_logits = resnet(norm(image))
        logits_uncompressed[idx] = ori_logits
        (target_class, target_dim) = return_class_name(id_classname_json, ori_logits[-1].unsqueeze(0))
        original_acc = return_class_accuracy(ori_logits[-1].unsqueeze(0), target_dim)
        ori_train_acc = ori_train_acc + (ori_logits.argmax(dim=1) == labels).sum()
ori_train_acc = ori_train_acc / len(train_dataset)        


for epoch in range(100):
    running_train_loss = 0.0
    running_train_acc = 0.0
    running_val_loss = 0.0
    running_val_acc = 0.0
    
    temperature = max(0.001, initial_temperature*np.exp(-temperature_anneal_rate*epoch))
    
    for (image, labels, idx) in tqdm(train_loader):
        
        # y_table = gumbel_softmax(y_table_1hot.view(1, -1, 255), temperature, True) * categorical_values_table
        # c_table = gumbel_softmax(c_table_1hot.view(1, -1, 255), temperature, True) * categorical_values_table
        
        y_table = torch.nn.functional.softmax(y_table_1hot.view(1, -1, max_q_value-1)/temperature, dim=2) * categorical_values_table
        c_table = torch.nn.functional.softmax(c_table_1hot.view(1, -1, max_q_value-1)/temperature, dim=2) * categorical_values_table
        
        y_table = y_table.sum(dim=2).reshape(8,8)
        c_table = c_table.sum(dim=2).reshape(8,8)

        # print(y_table)
        
        compressed_image = JPEGCompress(image, y_table, c_table)
        data = norm(compressed_image)
        logits = resnet(data)
        pred = logits.argmax(dim=1)
        
        ori_logits = logits_uncompressed[idx]
        
        loss_minimize = loss(logits, ori_logits) #we try to minimize this loss
        loss_maximize = (- qloss(y_table, ones_table) - qloss(c_table, ones_table))
        total_loss = (1-alpha)*loss_minimize + (alpha)* loss_maximize #total loss to be optimized

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # print((y_table_1hot == best_y_1hot).all())
        
        running_train_loss = running_train_loss + total_loss.detach().cpu() * image.shape[0]
        running_train_acc = running_train_acc + (pred == labels).sum()
            
    if running_train_loss/len(train_dataset) < best_loss:
        best_y_table = y_table
        best_c_table = c_table
        best_loss = running_train_loss/len(train_dataset)
        torch.save({"y_table" : y_table,
                   "c_table" : c_table,
                    "optimizer" : optimizer.state_dict(),
                   }, "best_ckpt.tar")
        
    if epoch % 10 == 0 and epoch > 0:
        torch.save({"y_table" : y_table,
           "c_table" : c_table,
            "optimizer" : optimizer.state_dict(),
           }, "epoch_{0}.tar".format(epoch))
        
    print("epoch {0}, training_loss: {1}, training_acc: {2}, ori_train_acc: {3}, logit loss: {4}, q table loss: {5}".format(
        epoch, 
        round(running_train_loss.item()/len(train_dataset), 5), 
        round(running_train_acc.item()/len(train_dataset)*100, 2),
        round(ori_train_acc.item()*100, 2),
        round(loss_minimize.item(), 5), 
        round(loss_maximize.item(), 5)
    ))
        
        
        
y_table = torch.nn.functional.softmax(y_table_1hot.view(1, -1, 127)/temperature, dim=2) * categorical_values_table
c_table = torch.nn.functional.softmax(c_table_1hot.view(1, -1, 127)/temperature, dim=2) * categorical_values_table

y_table = y_table.sum(dim=2).reshape(8,8)
c_table = c_table.sum(dim=2).reshape(8,8)

print("y_table\n", y_table)
print("c_table\n", c_table)


# Calculate accuracy if y_table and c_table were integers
y_table = y_table_1hot.argmax(dim=2)+1
c_table = c_table_1hot.argmax(dim=2)+1

train_acc = 0
for (image, labels) in tqdm(train_loader):
    with torch.no_grad():
        compressed_image = JPEGCompress(image, y_table, c_table)
        logits = resnet(norm(compressed_image))
        (target_class, target_dim) = return_class_name(id_classname_json, logits[-1].unsqueeze(0))
        acc = return_class_accuracy(logits[-1].unsqueeze(0), target_dim)
        train_acc = train_acc + (logits.argmax(dim=1) == labels).sum()

print("Accuracy after optimization:", train_acc/len(train_dataset))


# check file size
import os
root = r"../../data/imagenette2/train"
total_size_ori = 0

for folder in os.listdir(root):
    for file in os.listdir(os.path.join(root, folder)):
        total_size_ori += os.stat(os.path.join(root, folder, file)).st_size
        
print("total size originally:", total_size_ori)

qtable = {0: y_table.flatten().tolist(), 1: c_table.flatten().tolist()}
f_buffer = BytesIO()
total_size_aft = 0

for folder in os.listdir(root):
    for file in os.listdir(os.path.join(root, folder)):
        im.save(f_buffer, 'JPEG', qtables=qtable)
        total_size_aft += f_buffer.getbuffer().nbytes
        f_buffer.seek(0)
        f_buffer.truncate(0)

print("total size after:", total_size_aft)
print("compression_ratio:", round(total_size_aft/float(total_size_ori), 2))