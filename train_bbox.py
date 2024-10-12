import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader

import random
import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import math

from utils.logger import Logger

from utils.DataBuilderBBOX import DatasetBuilder
from models.MODEL import MODEL

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--log_root_dir', type=str, default='logs_train_bbox', help='root dir of log files')
parser.add_argument('--dataset_root_dir', type=str, default='./dataset_anonymous', help='root dir of dataset')
parser.add_argument('--correction_matrix_path', type=str, default='./init/correction_matrix.npy')

parser.add_argument('--train_people_ids', type=str, default='0,3,4,5,6,7,9,12,13,14', help='List of train people IDs separate with a single comma')
parser.add_argument('--test_people_ids', type=str, default='1,2,8,10,11', help='List of test people IDs separate with a single comma')
parser.add_argument('--device', type=str, default='0', help='training device')
parser.add_argument('--alpha', type=float, default=1.0)

parser.add_argument('--camera_jitter', type=int, default=0)
parser.add_argument('--noise', type=int, default=0)
parser.add_argument('--n_mask', type=int, default=0)

parser.add_argument('--num_epochs', type=int, default=300)
parser.add_argument('--height', type=int, default=480)
parser.add_argument('--width', type=int, default=640)

def setup_device(args):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = args.device
    if device == 'cpu':
        return device
    
    device = torch.device("cuda:"+args.device if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(seed)
    return device

def setup_dataloader(dataset_paths, batch_size, train_shuffle, train_people_ids, test_people_ids, args):
    train_dataset = DatasetBuilder(dataset_paths, train=True, train_people_ids=train_people_ids, test_people_ids=test_people_ids, noise=args.noise, n_mask=args.n_mask, camera_jitter=args.camera_jitter, height=args.height, width=args.width, correction_matrix_path=args.correction_matrix_path)
    test_dataset = DatasetBuilder(dataset_paths, train=False, train_people_ids=train_people_ids, test_people_ids=test_people_ids, noise=args.noise, n_mask=args.n_mask, camera_jitter=args.camera_jitter, height=args.height, width=args.width, correction_matrix_path=args.correction_matrix_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, test_dataset, train_loader, test_loader


def train(model, alpha, criterion, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        x_img0, x_img0_center, x_img1, x_img1_center, X_world_center, X_world_center_BEV, UWB  = [d.to(device) for d in data]
        optimizer.zero_grad()
        
        X_world_center_pred, x_img0_center_pred, x_img1_center_pred, P1, P2 = model(x_img0, x_img1)
        
        loss_X = criterion(X_world_center_pred, X_world_center)
        loss_x1 = criterion(x_img0_center_pred, x_img0_center)
        loss_x2 = criterion(x_img1_center_pred, x_img1_center)
        loss = loss_X + alpha*loss_x1 + alpha*loss_x2
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def compute_ate(est_trajectory, gt_trajectory):
    delta_trans = gt_trajectory[0, :] - est_trajectory[0, :]
    adjusted_est_trajectory = est_trajectory + delta_trans
    error = torch.sqrt(torch.sum((gt_trajectory - adjusted_est_trajectory)**2, dim=1))
    ate = torch.mean(error)
    return ate.item()

def compute_rpe(est_trajectory, gt_trajectory):
    gt_deltas = gt_trajectory[1:, :] - gt_trajectory[:-1, :]
    est_deltas = est_trajectory[1:, :] - est_trajectory[:-1, :]
    error = torch.sqrt(torch.sum((gt_deltas - est_deltas)**2, dim=1))
    rpe = torch.mean(error)
    return rpe.item()


@torch.no_grad()
def evaluate(model, alpha, criterion, data_loader, threshes, device, args):
    model.eval()
    metrics = {
        'loss': 0,
        'loss_X': 0,
        'loss_x1': 0,
        'loss_x2': 0,
        
        'error_X_mean': 0,
        'error_X_median': 0,
        'error_X_std': 0,

        'error_x1_mean': 0,
        'error_x1_median': 0,
        'error_x1_std': 0,

        'error_x2_mean': 0,
        'error_x2_median': 0,
        'error_x2_std': 0,
        
        'ATE_error': 0,
        'RPE_error': 0,
        'accuracy': {thresh: 0 for thresh in threshes}
    }
    
    X_outputs_list = []
    X_targets_list = []
    x1_outputs_list = []
    x1_targets_list = []
    x2_outputs_list = []
    x2_targets_list = []
    
    with torch.no_grad():
        for data in data_loader:
            x_img0, x_img0_center, x_img1, x_img1_center, X_world_center, X_world_center_BEV, UWB_BEV  = [d.to(device) for d in data]
            
            X_world_center_pred, x_img0_center_pred, x_img1_center_pred, P1, P2 = model(x_img0, x_img1)
            
            loss_X = criterion(X_world_center_pred, X_world_center)
            loss_x1 = criterion(x_img0_center_pred, x_img0_center)
            loss_x2 = criterion(x_img1_center_pred, x_img1_center)
            loss = loss_X + alpha*loss_x1 + alpha*loss_x2
            
            metrics['loss_X'] += loss_X.item()
            metrics['loss_x1'] += loss_x1.item()
            metrics['loss_x2'] += loss_x2.item()
            metrics['loss'] += loss.item()
            
            X_outputs_list.append(X_world_center_pred[:, :3].cpu())
            X_targets_list.append(X_world_center[:, :3].cpu())
            x1_outputs_list.append(x_img0_center_pred[:, :2].cpu())
            x1_targets_list.append(x_img0_center[:, :2].cpu())
            x2_outputs_list.append(x_img1_center_pred[:, :2].cpu())
            x2_targets_list.append(x_img1_center[:, :2].cpu())
    
    
    metrics['loss'] /= len(data_loader)
    metrics['loss_X'] /= len(data_loader)
    metrics['loss_x1'] /= len(data_loader)
    metrics['loss_x2'] /= len(data_loader)
    
    X_error_dis = F.pairwise_distance(torch.cat(X_outputs_list, 0), torch.cat(X_targets_list, 0), p=2)
    metrics['error_X_mean'] = X_error_dis.mean().item()
    metrics['error_X_median'] = X_error_dis.median().item()
    metrics['error_X_std'] = X_error_dis.std().item()
    for thresh in threshes:
        acc = (X_error_dis <= thresh).float().mean().item()
        metrics['accuracy'][thresh] = acc
        
    x1_outputs_tensor = torch.cat(x1_outputs_list, 0)
    x1_targets_tensor = torch.cat(x1_targets_list, 0)
    x1_outputs_tensor = ((x1_outputs_tensor + 1) / 2) * torch.tensor([args.width, args.height])
    x1_targets_tensor = ((x1_targets_tensor + 1) / 2) * torch.tensor([args.width, args.height])
    x1_error_dis = F.pairwise_distance(x1_outputs_tensor, x1_targets_tensor, p=2)
    metrics['error_x1_mean'] = x1_error_dis.mean().item()
    metrics['error_x1_median'] = x1_error_dis.median().item()
    metrics['error_x1_std'] = x1_error_dis.std().item()
    
    x2_outputs_tensor = torch.cat(x2_outputs_list, 0)
    x2_targets_tensor = torch.cat(x2_targets_list, 0)
    x2_outputs_tensor = ((x2_outputs_tensor + 1) / 2) * torch.tensor([args.width, args.height])
    x2_targets_tensor = ((x2_targets_tensor + 1) / 2) * torch.tensor([args.width, args.height])
    x2_error_dis = F.pairwise_distance(x2_outputs_tensor, x2_targets_tensor, p=2)
    metrics['error_x2_mean'] = x2_error_dis.mean().item()
    metrics['error_x2_median'] = x2_error_dis.median().item()
    metrics['error_x2_std'] = x2_error_dis.std().item()
    
    metrics['ATE_error'] = compute_ate(torch.cat(X_outputs_list, 0), torch.cat(X_targets_list, 0))
    metrics['RPE_error'] = compute_rpe(torch.cat(X_outputs_list, 0), torch.cat(X_targets_list, 0))
    
    return metrics

def display_acc(logger, train_metrics, test_metrics, train_metrics_history, test_metrics_history, epoch, threshes):
    logger.write(f"\nEpoch {epoch} - Train vs. Test\n")
    for key, value in train_metrics.items():
        if key != 'accuracy':
            logger.write(f"{key}: Train={value:.4f}, Test={test_metrics[key]:.4f}\n")
            if key not in train_metrics_history:
                train_metrics_history[key] = []
                test_metrics_history[key] = []
            train_metrics_history[key].append(value)
            test_metrics_history[key].append(test_metrics[key])
            
    for thresh in threshes:
        logger.write(f"Accuracy @ {thresh}m: Train={train_metrics['accuracy'][thresh]*100:.2f}%, Test={test_metrics['accuracy'][thresh]*100:.2f}%\n")
        key = f"Accuracy@{thresh}m"
        if key not in train_metrics_history:
            train_metrics_history[key] = []
            test_metrics_history[key] = []
        train_metrics_history[key].append(train_metrics['accuracy'][thresh])
        test_metrics_history[key].append(test_metrics['accuracy'][thresh])
        
    logger.flush()
    

def save_plot_metrics(train_metrics_history, test_metrics_history, vis_dir):
    for key in train_metrics_history:
        plt.figure(figsize=(10, 5))
        epochs = range(1, len(train_metrics_history[key]) + 1)
        plt.plot(epochs, train_metrics_history[key], label=f'train {key}')
        plt.plot(epochs, test_metrics_history[key], label=f'test {key}')
        plt.title(f'{key} Over Epochs')
        plt.xlabel('epoch')
        plt.ylabel(key)
        plt.legend()
        plt.savefig(os.path.join(vis_dir, f'{key}_over_epochs.png'))
        plt.close()
        
def save_csv(test_people_ids_str, train_people_ids_str, all_metrics, save_csv_path):
    rows = []
    for prefix, metrics in all_metrics:
        row = {
            'type': prefix,
            'test_people_ids': test_people_ids_str,
            'train_people_ids': train_people_ids_str
        }
        row.update({k: metrics[k] for k in metrics if k != 'accuracy'})
        row.update({f'acc@{thresh}': metrics['accuracy'][thresh] for thresh in metrics['accuracy']})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(save_csv_path, index=False)
    
def main(args):
    log_root_dir = args.log_root_dir
    alpha = args.alpha
    train_people_ids = [int(id) for id in args.train_people_ids.split(',')]
    test_people_ids = [int(id) for id in args.test_people_ids.split(',')]
    device = setup_device(args)
    

    batch_size = 128
    LR = 1e-3
    num_epochs = args.num_epochs

    num_kpt = 12
    model = MODEL(num_kpt=num_kpt).to(device)
    
    log_dir = os.path.join( log_root_dir, f'{str(model)}/alpha_{alpha}/{args.test_people_ids}/{args.train_people_ids}' )
    logger = Logger(os.path.join(log_dir, 'log.txt'))
    
    dataset_root_dir = args.dataset_root_dir
    
    walking_trainset, walking_testset, walking_trainloader, walking_testloader = setup_dataloader(
        dataset_paths = [ os.path.abspath(os.path.join(dataset_root_dir, 'walking', subdir, 'dataset.json')) for subdir in os.listdir(os.path.join(dataset_root_dir, 'walking'))],
        batch_size = batch_size,
        train_shuffle = True,
        train_people_ids=train_people_ids,
        test_people_ids=test_people_ids,
        args=args
    )
    
    cross_trainset, cross_testset, cross_trainloader, cross_testloader = setup_dataloader(
        dataset_paths = [ os.path.abspath(os.path.join(dataset_root_dir, 'cross', subdir, 'dataset.json')) for subdir in os.listdir(os.path.join(dataset_root_dir, 'cross'))],
        batch_size = batch_size,
        train_shuffle = False,
        train_people_ids=train_people_ids,
        test_people_ids=test_people_ids,
        args=args
    )
    square_trainset, square_testset, square_trainloader, square_testloader = setup_dataloader(
        dataset_paths = [ os.path.abspath(os.path.join(dataset_root_dir, 'square', subdir, 'dataset.json')) for subdir in os.listdir(os.path.join(dataset_root_dir, 'square'))],
        batch_size = batch_size,
        train_shuffle = False,
        train_people_ids=train_people_ids,
        test_people_ids=test_people_ids,
        args=args
    )
    logger.write(f"Using {device} device\n\n")
    logger.write(f"train_people_ids: {args.train_people_ids}\ntest_people_ids: {args.test_people_ids}\n\n")
    
    weights_dir = os.path.join(log_dir, 'weights')
    os.makedirs(weights_dir)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir)
    
    logger.write(f"alpha: {alpha}\n\n")
    len_train_data = len(walking_trainloader.dataset)
    len_test_data = len(walking_testloader.dataset) + len(cross_testloader.dataset) + len(square_testloader.dataset)+ len(cross_trainloader.dataset) + len(square_trainloader.dataset)
    logger.write(f"Train set size: {len_train_data}\n Test set size:\nWalking:{len(walking_testloader.dataset)}\n Cross_test:{len(cross_testloader.dataset)}\n Square_test:{len(square_testloader.dataset)}\n Cross_train:{len(cross_trainloader.dataset)}\n Square_train:{len(square_trainloader.dataset)}\n\n")

    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    threshes = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    best_epoch_info = {'epoch': 0, 'metrics': None}
    
    train_metrics_history = {}
    test_metrics_history = {}
    best_weight_path = ''
    for epoch in range(num_epochs):
        train_loss = train(model, alpha, criterion, optimizer, walking_trainloader, device)
        logger.write(f"\n\nEpoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}")

        train_metrics = evaluate(model, alpha, criterion, walking_trainloader, threshes, device, args)
        test_metrics = evaluate(model, alpha, criterion, walking_testloader, threshes, device, args)
        
        display_acc(logger, train_metrics, test_metrics, train_metrics_history, test_metrics_history, epoch, threshes)
        
        if best_epoch_info['metrics'] is None or test_metrics['error_X_mean'] < best_epoch_info['metrics']['error_X_mean']:
            best_epoch_info['epoch'] = epoch
            best_epoch_info['metrics'] = test_metrics
            best_weight_path = os.path.join(weights_dir, f"best_model_epoch.pth")
            torch.save(model.state_dict(), best_weight_path)

    logger.write("\n\nTraining completed\n\n")
    logger.write(f"Best Epoch: {best_epoch_info['epoch']}\n")
    for key, value in best_epoch_info['metrics'].items():
        if key != 'accuracy':
            logger.write(f"Best {key}: {value:.4f}\n")
        else:
            for thresh in threshes:
                logger.write(f"Best Accuracy @ {thresh}m: {value[thresh]*100:.2f}%\n")

    save_plot_metrics(train_metrics_history, test_metrics_history, vis_dir)
    
    model.load_state_dict(torch.load(best_weight_path))
    model.eval()
    walking_test_metrics = evaluate(model, alpha, criterion, walking_testloader, threshes, device, args)
    cross_train_metrics = evaluate(model, alpha, criterion, cross_trainloader, threshes, device, args)
    cross_test_metrics = evaluate(model, alpha, criterion, cross_testloader, threshes, device, args)
    square_train_metrics = evaluate(model, alpha, criterion, square_trainloader, threshes, device, args)
    square_test_metrics = evaluate(model, alpha, criterion, square_testloader, threshes, device, args)
    
    all_metrics = [
        
        ('walking_test', walking_test_metrics),
        ('cross_train', cross_train_metrics),
        ('cross_test', cross_test_metrics),
        ('square_train', square_train_metrics),
        ('square_test', square_test_metrics)

    ]

    save_csv_path = os.path.join( log_dir, 'table.csv' )
    save_csv(args.test_people_ids, args.train_people_ids, all_metrics, save_csv_path)
    logger.close()
    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)