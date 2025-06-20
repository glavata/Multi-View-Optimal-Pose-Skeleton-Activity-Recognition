import torch
import numpy as np
import time
import os
from pathlib import Path
from .hcn_model import HCN
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import tables
from tqdm import tqdm
from torch.utils.data import BatchSampler, SequentialSampler
import h5pickle as h5py
from torch.optim.lr_scheduler import ExponentialLR
from utils import logger

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

        len_act = int(self.dataset['seq_lens_act'][index])
        data_numpy = self.dataset['X_act'][index][None, :len_act, :, None]
        data_torch = torch.from_numpy(data_numpy).contiguous().to(torch.float32)
        data_torch = torch.permute(data_torch, (0, 2, 1, 3))
        data_torch = torch.nn.functional.interpolate(data_torch, size=(96, 1), mode='bilinear',align_corners=False).squeeze(dim=3).squeeze(dim=0)
        #data_torch = torch.permute(data_torch, (1, 0))
        data_torch = torch.reshape(data_torch, (25, 3, 96, 1))
        #N, C, T, V, M 

        #(C, V, M, T).permute(0, 3, 1, 2) -> C, T, V, M -> 3, 100, 25, 1
        data_torch = torch.permute(data_torch, (1, 2, 0, 3))#.astype(torch.float32)
        label = self.dataset['y_act'][index,0].astype(np.int64)
        
        return data_torch, label
    

def eval(loader, model, device, nb_classes=None):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    if(nb_classes is not None):
        confusion_matrix = torch.zeros(nb_classes, nb_classes)

    results = None

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            if(nb_classes is None):
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)
            else:
                for t, p in zip(y.view(-1), predictions.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        if(nb_classes is None):
            final_acc = float(num_correct)/float(num_samples)*100
            print(f'Got {num_correct} / {num_samples} with accuracy {final_acc:.2f}%') 
            results = final_acc
        else:
            num_correct = confusion_matrix.diag().sum()
            num_samples = confusion_matrix.sum()
            final_acc = num_correct / num_samples *100
            
            avg_recall = torch.mean(confusion_matrix.diag() / confusion_matrix.sum(axis=1)) * 100
            avg_precision = torch.mean(confusion_matrix.diag() / confusion_matrix.sum(axis=0)) * 100

            print(f'Got {num_correct} / {num_samples} with accuracy {final_acc:.2f}%, avg recall {avg_recall:.2f}%, avg precision {avg_precision:.2f}%')

            results = final_acc, confusion_matrix


    return results

def train_one_epoch(training_loader, model, optimizer, loss_fn, device, epoch_index, tb_writer):

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    loss_value = []
    acc_value = []
    dataset_len_tr = len(training_loader)

    model.train()

    process = tqdm(training_loader, total=dataset_len_tr)
    for i, data in enumerate(process):
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
        loss_value.append(loss.data.item())

        # Adjust learning weights
        optimizer.step()
        value, predict_label = torch.max(outputs.data, 1)
        acc = torch.mean((predict_label == labels).float())
        acc_value.append(acc.data.item())

        # Gather data and report

        tb_writer.add_scalar('Acc/train', acc, (epoch_index * dataset_len_tr) + i)
        tb_writer.add_scalar('Loss/train', loss.item(),  (epoch_index * dataset_len_tr) + i)

        lr = optimizer.param_groups[0]['lr']
        tb_writer.add_scalar('Learning rate/train', lr, (epoch_index * dataset_len_tr) + i)
        process.set_description('Loss: {:.4f}, LR: {:.4f}'.format(loss.data.item(), lr))

    print('\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(acc_value)*100))

def train_hcn(dataset_tr, dataset_ts, num_classes, merged_str, args):


    epochs, batch_size, base_lr, steps, save_freq = args['epochs'], args['batch_size'], args['base_lr'], args['steps'], \
                                                    args['save_freq']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_feeder_tr = Feeder(dataset_tr, device)
    dataset_feeder_ts = Feeder(dataset_ts, device)


    data_loader_tr = torch.utils.data.DataLoader(dataset=dataset_feeder_tr,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                #pin_memory=True,
                                                num_workers=4)
    
    data_loader_ts = torch.utils.data.DataLoader(dataset=dataset_feeder_ts,
                                                batch_size=batch_size,
                                                #shuffle=True,
                                                #sampler = sampler_ts,
                                                num_workers=4)
    
    model  = HCN(num_class=num_classes, window_size=96)

    #model.apply(weights_init)

    checkpoints_dir = str(Path('.') / "results" / "checkpoints")
    #model = torch.load(checkpoints_dir + os.sep + "HCN_model_weights_ntu_200_data_Non_Merged_epoch.pth")

    model.to(device)
    #check_accuracy(data_loader_ts, model, device)


    dataset_len_tr = len(data_loader_tr)      
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= base_lr, weight_decay= 1e-4, betas=(0.9, 0.999), eps=1e-8)
    scheduler =  ExponentialLR(optimizer, gamma=0.99, last_epoch=-1)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logs_dir = str(Path('.') / "results" / "logs") 
    writer = SummaryWriter(f'{logs_dir}{os.sep}{timestamp}')

    train_time_start = time.time()

    last_acc = 0
    epoch_cur = 0
    stop = False

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))
        scheduler.step()
        
        train_one_epoch(data_loader_tr, model, optimizer, loss_fn, device, epoch, writer)

        res_acc = eval(data_loader_ts, model, device)

        epoch_cur = epoch + 1

        if(stop):
            break
        
        if((epoch + 1) % save_freq == 0):
            print("{0}th epoch, saving checkpoint...".format(epoch + 1))
            torch.save(model, checkpoints_dir + os.sep + "HCN_model_weights_ntu_{0}_data_{1}_epoch.pth".format(merged_str, epoch + 1))

            if(stop):
                print("No improvement of validation accuracy, ending training...")
                break

        last_acc = res_acc

    train_time_end = time.time()
    print(" Training time for HCN with {0} epochs and {1} data: {2} seconds".format(epoch_cur, merged_str, train_time_end - train_time_start))

    final_acc, conf_mat = eval(data_loader_ts, model, device, num_classes)
    str_res = merged_str + "_HCN_" + str(epoch_cur) + "_epochs"
    logger.save_acc_prec_rec(str_res, final_acc, conf_mat.numpy())

    #torch.save(model, checkpoints_dir + os.sep + "HCN_model_weights_ntu_{0}_data_{1}_epoch.pth".format(merged_str, epoch_cur))


