import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.transforms import functional as F

from tv_dataset import TrainDataset

import random
import numpy as np

# Define the function to train an object detection model

# default parameters
num_classes=2; num_epochs=10; batch_size=4; learning_rate=0.005

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
g = torch.Generator()
g.manual_seed(0)



def train_object_detection_model(train_ds, num_classes=2, num_epochs=10, batch_size=4, learning_rate=0.005):
    """
    Trains a Faster R-CNN object detection model using the given train_ds.
    
    Args:
        train_ds: A PyTorch Dataset object where each sample is a dict with 'image' and 'target'.
                 'image' is a color image tensor, and 'target' is a dict with 'boxes' and 'labels'.
        num_classes: Number of classes (including background) in the train_ds.
        num_epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate for the optimizer.

    Returns:
        model: The trained Faster R-CNN model.
    """
    # Use the Faster R-CNN pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Replace the classifier head with a new one (for the number of classes)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # DataLoader to batch and shuffle the train_ds
    data_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=4, shuffle=False, worker_init_fn=seed_worker, generator=g, collate_fn=lambda x: tuple(zip(*x))) # on linux
    # n_batches = len(train_dl)



    # Define the optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Enable the model to run on GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for images, targets in data_loader:
            # Move images and targets to the correct device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        
        # Step the learning rate scheduler
        lr_scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(data_loader)}')

    return model


train_ds = TrainDataset(root='cellpose_train_1.png')

train_object_detection_model(train_ds)