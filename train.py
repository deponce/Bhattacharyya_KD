import argparse
import torch
import os
from tqdm import tqdm
import torch.optim as optim
from models import Model
from dataloader import get_loader
from utils import GetModel
from KD.knowledgedistillation import KnowledgeDistilldation
from torchvision.transforms import transforms
from torchvision.datasets.cifar import CIFAR100
from torch.utils.data import DataLoader
import torch.nn.functional as F
def loss_kd(preds, labels, teacher_preds, params):
    T = params.temperature
    alpha = params.alpha
    loss = T * T * alpha* F.kl_div(F.log_softmax(preds / T, dim=1), 
                                   F.softmax(teacher_preds / T, dim=1), reduction='batchmean') + \
                                \
             (1. - alpha)*F.cross_entropy(preds, labels)
    return loss

def HardLoss(preds, traget, T=4):

def main(args):
    Teacher = GetModel(args.teacher_name)
    Student = GetModel(args.model_name)
    if not args.device:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    Optimizer = optim.SGD(Student.parameters(), lr=args.lr, momentum=args.momentum)
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = CIFAR100('./dataset', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, args.batchsize, shuffle=True)
    val_dataset = CIFAR100('./dataset', train=False, transform=transform, download=True)
    val_loader = DataLoader(val_dataset, args.batchsize, shuffle=False)
    LossFn = []
    LossIndicator=["H", "S"]
    Lambda=[0.5, 0.5]
    KD = KnowledgeDistilldation(Teacher=Teacher, Student=Student, Optimizer=Optimizer, 
                                TrainLoader=train_loader, ValLoader=val_loader, LossFn=LossFn, 
                                LossIndicator=LossIndicator, Lambda=Lambda, device=device)
    for e in range(args.num_epoch):
        TrainResult = KD.TrainOneEpoch()
        ValResult = KD.Validation()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument('--params_dir', type=str, default="params", help='the directory of hyper parameters')
    parser.add_argument('-m', '--model_name', type=str, default='base', help='the name of backbone network')
    parser.add_argument('-t', '--teacher_name', type=str, default=None, help='the name of backbone network')
    parser.add_argument('--lr', type=float, default=0.5, help='learning rate')
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

    # student_params = Params(os.path.join(args.params_dir, f'{args.model_name}.json'))
    # if args.teacher_name is None:
    #     teacher_params = Params(os.path.join(args.params_dir, f'{student_params.teacher_name}.json'))
    # else:
    #     teacher_params = Params(os.path.join(args.params_dir, f'{args.teacher_name}.json'))

    # student = Model(args.num_classes, student_params, args.epoch)
    # student.load_params(os.path.join(args.checkpoint_dir, args.dataset, student_params.model_name, f'{args.epoch-1}.pth'))

    # teacher = Model(args.num_classes, teacher_params)
    # teacher.load_params(os.path.join(args.checkpoint_dir, args.dataset, teacher_params.model_name, f'final.pth'))

    # summary_title = f'{student_params.teacher_name}_teaches_{student_params.model_name} '

    # if not os.path.exists(os.path.join(args.checkpoint_dir, args.dataset, student_params.model_name)):
    #     os.makedirs(os.path.join(args.checkpoint_dir, args.dataset, student_params.model_name))
    # if not os.path.exists(args.log_path):
    #     os.makedirs(args.log_path)
    # writer = SummaryWriter(args.log_path)

    # criterion = loss_kd
    # optimizer = torch.optim.Adam(student.parameters(), student_params.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_epoch)
    # train_loader, validation_loader = get_loader(args.image_size, student_params.batch_size, args.dataset)
    # scheduler.step(args.epoch)

    # teacher_train_ans = teacher.fetch_output(train_loader)
    # teacher_val_ans = teacher.fetch_output(validation_loader)

    # for iter in range(args.epoch, args.num_epoch):
    #     train_loss, train_acc = student.train_model(train_loader, criterion, optimizer, teacher_train_ans, student_params)
    #     validation_loss, validation_acc = student.validate_model(validation_loader, criterion, teacher_val_ans, student_params)
    #     writer.add_scalars(f'{summary_title}/Loss', {'train': train_loss, 'val': validation_loss}, iter)
    #     writer.add_scalars(f'{summary_title}/Accuracy', {'train': train_acc, 'val': validation_acc}, iter)
    #     torch.save(student.state_dict(), os.path.join(args.checkpoint_dir, args.dataset, student_params.model_name, f'{iter}.pth'))
    #     scheduler.step()
