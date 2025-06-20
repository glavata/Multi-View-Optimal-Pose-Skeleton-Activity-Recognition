import torch
import numpy as np
import time
import os
from pathlib import Path
from .st_gcn_model import ST_GCN_18
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import tables
from tqdm import tqdm
from torch.utils.data import BatchSampler, SequentialSampler
import h5pickle as h5py

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv1d') != -1:
        model.weight.data.normal_(0.0, 0.02)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        model.weight.data.normal_(0.0, 0.02)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)


class Feeder(torch.utils.data.Dataset):

    def __init__(self, filename, device):

        self.file_path = filename
        self.device = device
        file = h5py.File(self.file_path, mode='r')
        self.dataset = file
        self.dataset_len = file['y_act'].shape[0]


    # def open_hdf5(self):


    def __del__(self):
        if hasattr(self, 'dataset'):
            self.dataset.close()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        # get data
        # if not hasattr(self, 'dataset'):
        #     self.open_hdf5()

        
        data_numpy = np.reshape(self.dataset['X_act'][index], (300, 25, 3, 1))
        data_numpy = np.transpose(data_numpy, (2, 0, 1, 3)).astype(np.float32) #N, C, T, V, M -> batch_size, 3, 300, 25, 1
        label = self.dataset['y_act'][index,0].astype(np.int64)
        
        return data_numpy, label
    



def train_gcn(dataset_tr, dataset_ts, num_classes, merged_str, args):


    epochs, batch_size, base_lr, steps, save_freq = args['epochs'], args['batch_size'], args['base_lr'], args['steps'], \
                                                    args['save_freq']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_feeder_tr = Feeder(dataset_tr, device)
    dataset_feeder_ts = Feeder(dataset_ts, device)


    data_loader_tr = torch.utils.data.DataLoader(dataset=dataset_feeder_tr,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                #pin_memory=True,
                                                num_workers=4)
    
    data_loader_ts = torch.utils.data.DataLoader(dataset=dataset_feeder_ts,
                                                batch_size=batch_size,
                                                #shuffle=True,
                                                #sampler = sampler_ts,
                                                num_workers=4)
    
    model  = ST_GCN_18(in_channels=3,
                        num_class=num_classes,
                        dropout = 0.5,
                        layout= 'ntu-rgb+d',
                        strategy= 'spatial',
                        edge_importance_weighting=True)
    model.apply(weights_init)

    checkpoints_dir = str(Path('.') / "results" / "checkpoints")
    #model = torch.load(checkpoints_dir + os.sep + "model_weights_ntu_merged_data_40_epochs.pth")

    model.to(device)
    dataset_len_tr = len(data_loader_tr)      
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr= base_lr, momentum= 0.9, nesterov= True, weight_decay= 0.0001)

    def check_accuracy(loader, model):
        num_correct = 0
        num_samples = 0
        model.eval()
        
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                
                scores = model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)
            
            final_acc = float(num_correct)/float(num_samples)*100
            print(f'Got {num_correct} / {num_samples} with accuracy {final_acc:.2f}%') 

        return final_acc

    def train_one_epoch(training_loader, epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in tqdm(enumerate(training_loader), total=dataset_len_tr):
            # Every data instance is an input + label pair
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss


    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logs_dir = str(Path('.') / "results" / "logs") 
    writer = SummaryWriter(f'{logs_dir}{os.sep}{timestamp}')

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(data_loader_tr, epoch, writer)


        #running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        #model.eval()


        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',{ 'Training' : avg_loss}, epoch + 1)
        writer.flush()

    test_acc = check_accuracy(data_loader_ts, model)
    print(1)


