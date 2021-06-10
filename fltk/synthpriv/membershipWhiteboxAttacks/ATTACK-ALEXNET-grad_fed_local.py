import errno
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
if use_cuda:
    torch.cuda.empty_cache()
    torch.cuda.manual_seed_all(manualSeed)


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model


class AverageMeter(object):
    """Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class NasrAttackV2(nn.Module):
    def __init__(self, num_classes, num_models):
        self.num_classes = num_classes
        self.num_models = num_models
        super(NasrAttackV2, self).__init__()
        self.grads_conv = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv2d(1, 1000, kernel_size=(1, 100), stride=1),
            nn.ReLU(),
        )
        self.grads_linear = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(256 * 1000, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.labels = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.preds = nn.Sequential(
            nn.Linear(num_classes, 100),
            nn.ReLU(),
            nn.Linear(100, 64),
            nn.ReLU(),
        )
        self.correct = nn.Sequential(
            nn.Linear(1, num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, 64),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear((64 + 64 + 64 + 128) * self.num_models, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.output = nn.Sigmoid()

        for key in self.state_dict():
            if key.split(".")[-1] == "weight":
                nn.init.normal_(self.state_dict()[key], std=0.01)
            elif key.split(".")[-1] == "bias":
                self.state_dict()[key][...] = 0

    def forward(self, gs, l, cs, os):
        for i in range(self.num_models):
            out_g = self.grads_linear(self.grads_conv(gs[i]).view([gs[i].size()[0], -1]))
            out_l = self.labels(l)  # labels are the same for all target models
            out_c = self.correct(cs[i])
            out_o = self.preds(os[i])
            if i == 0:
                com_inp = torch.cat((out_g, out_l, out_c, out_o), 1)
            else:
                com_inp = torch.cat((out_g, out_l, out_c, out_o, com_inp), 1)
        is_member = self.combine(com_inp)
        return self.output(is_member)


def mia(
    memloader,
    nonmemloader,
    attack_model,
    criterion,
    optimizer,
    classifiers,
    classifier_criterion,
    classifier_optimizers,
    num_batches=10,
):
    losses, top1 = AverageMeter(), AverageMeter()

    attack_model.train() if optimizer else attack_model.eval()
    for model in classifiers:
        model.eval()

    data_iter = iter(zip(memloader, nonmemloader))

    for _ in range(num_batches):
        (mem_input, mem_target), (nonmem_input, nonmem_target) = next(data_iter)
        mem_input, nonmem_input = mem_input.cuda(), nonmem_input.cuda()
        mem_target, nonmem_target = mem_target.cuda(), nonmem_target.cuda()

        model_input = torch.cat((mem_input, nonmem_input))
        pred_outputs = [classifier(model_input) for classifier in classifiers]

        labels = torch.cat((mem_target, nonmem_target))
        labels_1hot = nn.functional.one_hot(labels, num_classes=num_classes).float()

        classifiers_outputs, correct_labels, model_grads = [], [], []
        for m_n in range(len(classifiers)):
            correct = torch.sum(pred_outputs[m_n] * labels_1hot, dim=1)
            grads = torch.zeros(0)

            for i in range(2 * batch_size):
                loss_classifier = classifier_criterion(pred_outputs[m_n][i].view([1, -1]), labels[i].view([-1]))
                classifier_optimizers[m_n].zero_grad()
                if i == (2 * batch_size) - 1:
                    loss_classifier.backward(retain_graph=False)
                else:
                    loss_classifier.backward(retain_graph=True)
                g = classifiers[m_n].classifier.weight.grad.view([1, 1, 256, 100])

                if grads.size()[0] != 0:
                    grads = torch.cat((grads, g))
                else:
                    grads = g

            classifiers_outputs.append(pred_outputs[m_n].detach())
            correct_labels.append(correct.view([-1, 1]).detach())
            model_grads.append(grads.detach())

        attack_output = attack_model(model_grads, labels_1hot, correct_labels, classifiers_outputs)
        is_member = torch.cat((torch.zeros(batch_size, device="cuda"), torch.ones(batch_size, device="cuda")))[:, None]

        if optimizer:
            loss = criterion(attack_output, is_member)
            losses.update(loss.detach().item(), model_input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        prec1 = torch.mean(((attack_output > 0.5) == is_member).float()).detach().item()
        top1.update(prec1, model_input.size(0))

    return losses.avg, top1.avg


if __name__ == "__main__":
    mp.set_start_method("spawn")
    mp.set_sharing_strategy("file_system")

    dataset_name = "cifar100"
    checkpoint_path = "checkpoints_100cifar_alexnet_white_fed"
    lr = 0.05
    num_models = 1
    epochs = 50

    # Data
    print("==> Preparing dataset %s" % dataset_name)

    if dataset_name == "cifar10":
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    trainset = dataloader(root="./data100", train=True, download=True, transform=transforms.ToTensor())
    trainloader = data.DataLoader(trainset, batch_size=100, shuffle=True)

    testset = dataloader(root="./data100", train=False, download=True, transform=transforms.ToTensor())
    testloader = data.DataLoader(testset, batch_size=100, shuffle=True)

    trainset_member, testset_member = [], []
    r = np.random.permutation(50000)
    for i in range(25000):
        trainset_member.append(trainset[r[i]])
    for i in range(25000, 50000):
        testset_member.append(trainset[r[i]])

    trainset_nonmember, testset_nonmember = [], []
    r = np.random.permutation(10000)
    for i in range(5000):
        trainset_nonmember.append(testset[r[i]])
    for i in range(5000, 10000):
        testset_nonmember.append(testset[r[i]])

    batch_size = 20

    trainloader_member = data.DataLoader(trainset_member, batch_size=batch_size, shuffle=True)
    trainloader_nonmember = data.DataLoader(trainset_nonmember, batch_size=batch_size, shuffle=True)

    testloader_member = data.DataLoader(testset_member, batch_size=batch_size, shuffle=True)
    testloader_nonmember = data.DataLoader(testset_nonmember, batch_size=batch_size, shuffle=True)

    # load models for membership inference
    print("==> Loading models")

    criterion = nn.CrossEntropyLoss()

    nets = []
    optimizers = []
    for _ in range(num_models):
        net = AlexNet(num_classes)
        net = net.cuda()
        checkpoint = torch.load("checkpoints_100cifar_alexnet_white_fed/epoch_%d_main" % 299)
        net.load_state_dict(checkpoint["state_dict"])
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)

        nets.append(net)
        optimizers.append(optimizer)

    # create attack model
    criterion_attack = nn.MSELoss()
    attack_model = NasrAttackV2(100, 1)
    attack_model = attack_model.cuda()
    optimizer_attack = optim.Adam(attack_model.parameters(), lr=0.0001)

    print("==> Training attack")
    with tqdm(range(epochs)) as pbar:
        for epoch in pbar:
            loss, train_acc = mia(
                memloader=trainloader_member,
                nonmemloader=trainloader_nonmember,
                attack_model=attack_model,
                criterion=criterion_attack,
                optimizer=optimizer_attack,
                classifier_criterion=criterion,
                classifiers=nets,
                classifier_optimizers=optimizers,
            )
            pbar.write(f"Loss: {loss:.4f} | Train Accuracy: {train_acc: .4f}")
            if (epoch + 1) % 5 == 0:
                _, test_acc = mia(
                    memloader=testloader_member,
                    nonmemloader=testloader_nonmember,
                    attack_model=attack_model,
                    criterion=criterion_attack,
                    optimizer=None,  # no attack optimizer => runs test without optimizing
                    classifier_criterion=criterion,
                    classifiers=nets,
                    classifier_optimizers=optimizers,
                    num_batches=100,
                )
                pbar.write(f"Test accuracy: {test_acc:.4f}")

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "state_dict": attack_model.state_dict(),
                        "optimizer": optimizer_attack.state_dict(),
                    },
                    f"{checkpoint_path}/nasr_attack_model_{dataset_name}_{nets[0].__class__.__name__}_{epoch+1}.pt",
                )