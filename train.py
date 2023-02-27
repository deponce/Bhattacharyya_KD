import torch
import os
from tqdm import tqdm
import torch.optim as optim
# from models import Model
# from dataloader import get_loader
from utils import GetModel
from loss import SoftLoss
from KD.knowledgedistillation import KnowledgeDistilldation
from torchvision.transforms import transforms
from torchvision.datasets.cifar import CIFAR100
from torch.utils.data import DataLoader
import torch.nn.functional as F

def main(args):
    if not args.device:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    Teacher = GetModel(args.teacher_name, pretrained=True).to(device)
    Student = GetModel(args.model_name, pretrained=False).to(device)
    Optimizer = optim.SGD(Student.parameters(), lr=args.lr, momentum=args.momentum)
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = CIFAR100('./dataset', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, args.batchsize, shuffle=True)
    val_dataset = CIFAR100('./dataset', train=False, transform=transform, download=True)
    val_loader = DataLoader(val_dataset, args.batchsize, shuffle=False)
    softloss = SoftLoss(T=4)
    LossFn = [F.cross_entropy, softloss]
    LossIndicator=["H", "S"]
    Lambda=[1, 0]
    KD = KnowledgeDistilldation(Teacher=Teacher, Student=Student, Optimizer=Optimizer, \
                                TrainLoader=train_loader, ValLoader=val_loader, LossFn=LossFn,\
                                LossIndicator=LossIndicator, Lambda=Lambda, device=device)
    for epoch in range(args.num_epoch):
        TrainResult = KD.TrainOneEpoch()
        ValResult = KD.Validation()
        print(epoch, TrainResult['Top1'])
    return 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network')
    # parser.add_argument('--params_dir', type=str, default="params", help='the directory of hyper parameters')
    parser.add_argument('--device', type=str, default='cpu', help='cuda or cpu')
    parser.add_argument('--model_name', type=str, default='base', help='the name of backbone network')
    parser.add_argument('--teacher_name', type=str, default=None, help='the name of backbone network')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--params_dir', type=str, default="params", help='the directory of hyper parameters')
    parser.add_argument('--log_path', type=str, default='logs', help="directory to save train log")
    parser.add_argument('--epoch', type=int, default=0, help='value of current epoch')
    parser.add_argument('--batchsize', type=int, default=128, help='value of current epoch')
    parser.add_argument('--num_epoch', type=int, default=90, help='the number of epoch in train')
    parser.add_argument('--decay_epoch', type=int, default=30, help='the number of decay epoch in train')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="path to saved models (to continue training)")
    parser.add_argument('--num_classes', type=int, default=100, help='the number of classes')
    parser.add_argument('--dataset', type=str, default='cifar100', help='the name of dataset')
    parser.add_argument('--is_distill', type=bool, default=True)
    args = parser.parse_args()
    main(args)
