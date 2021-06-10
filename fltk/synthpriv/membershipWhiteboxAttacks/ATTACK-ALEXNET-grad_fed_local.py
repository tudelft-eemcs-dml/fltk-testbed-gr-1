from __future__ import print_function

import argparse
import os
import pickle
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

from utils import AverageMeter, accuracy, mkdir_p

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


class InferenceAttack(nn.Module):
    def __init__(self):
        super(InferenceAttack, self).__init__()

        self.features = nn.Sequential(nn.Linear(100, 64), nn.ReLU(), nn.Linear(64, 1))
        for key in self.state_dict():
            if key.split(".")[-1] == "weight":
                nn.init.normal_(self.state_dict()[key], std=0.01)
                print(key)

            elif key.split(".")[-1] == "bias":
                self.state_dict()[key][...] = 0
        self.output = nn.Sigmoid()

    def forward(self, x):
        is_member = self.features(x)

        return self.output(is_member)


class InferenceAttack_HZ(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super(InferenceAttack_HZ, self).__init__()
        # self.grads_conv = nn.Sequential(
        #     nn.Dropout(p=0.2),
        #     nn.Conv2d(1, 1000, kernel_size=(1, 100), stride=1),
        #     nn.ReLU(),
        # )
        self.grads_linear = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(256 * 100, 1024),
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
            nn.Linear(64 + 64 + 64 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        for key in self.state_dict():
            if key.split(".")[-1] == "weight":
                nn.init.normal_(self.state_dict()[key], std=0.01)
            elif key.split(".")[-1] == "bias":
                self.state_dict()[key][...] = 0
        self.output = nn.Sigmoid()

    def forward(self, g, l, c, o):
        # out_g = self.grads_conv(g).view([g.size()[0],-1])
        out_g = self.grads_linear(g.view([g.size()[0], -1]))
        out_l = self.labels(l)
        out_c = self.correct(c)
        out_o = self.preds(o)
        is_member = self.combine(torch.cat((out_g, out_c, out_l, out_o), 1))
        return self.output(is_member)


class InferenceAttack_HZ_FED(nn.Module):
    def __init__(self, num_classes, num_feds):
        self.num_classes = num_classes
        self.num_feds = num_feds
        super(InferenceAttack_HZ_FED, self).__init__()
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
            nn.Linear(64 * 4 * self.num_feds, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        for key in self.state_dict():
            # print(key)
            if key.split(".")[-1] == "weight":
                nn.init.normal_(self.state_dict()[key], std=0.01)
                # print(key)

            elif key.split(".")[-1] == "bias":
                self.state_dict()[key][...] = 0
        self.output = nn.Sigmoid()

    def forward(self, gs, ls, cs, os):

        for i in range(self.num_feds):
            out_g = self.grads_conv(gs[i]).view([gs[i].size()[0], -1])
            out_g = self.grads_linear(out_g)
            out_l = self.labels(ls[i])
            out_c = self.correct(cs[i])
            out_o = self.preds(os[i])
            if i == 0:
                com_inp = torch.cat((out_g, out_c, out_o), 1)
            else:
                com_inp = torch.cat((out_g, out_c, out_o, com_inp), 1)

        is_member = self.combine(com_inp)

        return self.output(is_member)


def train(trainloader, model, criterion, optimizer, pbar, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data, inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    pbar.write(
        "Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}".format(
            data=data_time.avg, bt=batch_time.avg, loss=losses.avg, top1=top1.avg, top5=top5.avg
        )
    )

    return (losses.avg, top1.avg)


def test(testloader, model, criterion, pbar, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data, inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    pbar.write(
        "Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}".format(
            data=data_time.avg,
            bt=batch_time.avg,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
    )

    return (losses.avg, top1.avg)


def mia_train_fed(
    trainloader,
    testloader,
    models,
    inference_model,
    classifier_criterion,
    criterion,
    classifier_optimizers,
    optimizer,
    pbar,
    num_batchs=1000,
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    mtop1_a = AverageMeter()
    mtop5_a = AverageMeter()

    inference_model.train()
    for model in models:
        model.eval()
    # switch to evaluate mode

    end = time.time()
    for batch_idx, ((mem_input, mem_target), (nonmem_input, nonmem_target)) in enumerate(zip(trainloader, testloader)):
        # measure data loading time
        if batch_idx > num_batchs:
            break
        data_time.update(time.time() - end)
        mem_input = mem_input.cuda()
        nonmem_input = nonmem_input.cuda()
        mem_target = mem_target.cuda()
        nonmem_target = nonmem_target.cuda()

        v_mem_input = torch.autograd.Variable(mem_input)
        v_nonmem_input = torch.autograd.Variable(nonmem_input)
        v_mem_target = torch.autograd.Variable(mem_target)
        v_nonmem_target = torch.autograd.Variable(nonmem_target)

        # compute output
        model_input = torch.cat((v_mem_input, v_nonmem_input))
        pred_outputs = []
        for i in range(len(models)):
            pred_outputs.append(models[i](model_input))

        infer_input = torch.cat((v_mem_target, v_nonmem_target))

        one_hot_tr = torch.from_numpy((np.zeros((infer_input.size(0), 100)))).cuda().type(torch.cuda.FloatTensor)
        target_one_hot_tr = one_hot_tr.scatter_(1, infer_input.type(torch.cuda.LongTensor).view([-1, 1]).data, 1)
        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)

        models_outputs = []
        correct_labels = []
        model_grads = []

        for m_n in range(len(models)):

            correct = torch.sum(pred_outputs[m_n] * infer_input_one_hot, dim=1)
            grads = torch.zeros(0)

            for i in range(2 * batch_size):
                loss_classifier = classifier_criterion(pred_outputs[m_n][i].view([1, -1]), infer_input[i].view([-1]))
                classifier_optimizers[m_n].zero_grad()
                if i == (2 * batch_size) - 1:
                    loss_classifier.backward(retain_graph=False)
                else:
                    loss_classifier.backward(retain_graph=True)
                g = models[m_n].classifier.weight.grad.view([1, 1, 256, 100])

                if grads.size()[0] != 0:

                    grads = torch.cat((grads, g))

                else:
                    grads = g

            grads = torch.autograd.Variable(torch.from_numpy(grads.data.cpu().numpy()).cuda())
            c = torch.autograd.Variable(torch.from_numpy(correct.view([-1, 1]).data.cpu().numpy()).cuda())
            preds = torch.autograd.Variable(torch.from_numpy(pred_outputs[m_n].data.cpu().numpy()).cuda())
            models_outputs.append(preds)
            correct_labels.append(c)
            model_grads.append(grads)
        member_output = inference_model(model_grads, infer_input_one_hot, correct_labels, models_outputs)

        is_member_labels = torch.from_numpy(
            np.reshape(np.concatenate((np.zeros(v_mem_input.size(0)), np.ones(v_nonmem_input.size(0)))), [-1, 1])
        ).cuda()

        v_is_member_labels = torch.autograd.Variable(is_member_labels).type(torch.cuda.FloatTensor)

        loss = criterion(member_output, v_is_member_labels)

        # measure accuracy and record loss
        prec1 = np.mean((member_output.data.cpu().numpy() > 0.5) == v_is_member_labels.data.cpu().numpy())
        losses.update(loss.data, model_input.size(0))
        top1.update(prec1, model_input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if batch_idx % 10 == 0:
            pbar.write(
                "({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} ".format(
                    batch=batch_idx,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                )
            )

    return (losses.avg, top1.avg)


def mia_test_fed(
    trainloader,
    testloader,
    models,
    inference_model,
    classifier_criterion,
    criterion,
    classifier_optimizers,
    pbar,
    num_batchs=1000,
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    mtop1_a = AverageMeter()
    mtop5_a = AverageMeter()

    inference_model.eval()
    for model in models:
        model.eval()
    # switch to evaluate mode

    end = time.time()
    for batch_idx, ((mem_input, mem_target), (nonmem_input, nonmem_target)) in enumerate(zip(trainloader, testloader)):
        # measure data loading time
        if batch_idx > num_batchs:
            break
        data_time.update(time.time() - end)
        mem_input = mem_input.cuda()
        nonmem_input = nonmem_input.cuda()
        mem_target = mem_target.cuda()
        nonmem_target = nonmem_target.cuda()

        v_mem_input = torch.autograd.Variable(mem_input)
        v_nonmem_input = torch.autograd.Variable(nonmem_input)
        v_mem_target = torch.autograd.Variable(mem_target)
        v_nonmem_target = torch.autograd.Variable(nonmem_target)

        # compute output
        model_input = torch.cat((v_mem_input, v_nonmem_input))
        pred_outputs = []
        for i in range(len(models)):
            pred_outputs.append(models[i](model_input))

        infer_input = torch.cat((v_mem_target, v_nonmem_target))

        one_hot_tr = torch.from_numpy((np.zeros((infer_input.size(0), 100)))).cuda().type(torch.cuda.FloatTensor)
        target_one_hot_tr = one_hot_tr.scatter_(1, infer_input.type(torch.cuda.LongTensor).view([-1, 1]).data, 1)
        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)

        models_outputs = []
        correct_labels = []
        model_grads = []

        for m_n in range(len(models)):

            correct = torch.sum(pred_outputs[m_n] * infer_input_one_hot, dim=1)
            grads = torch.zeros(0)

            for i in range(2 * batch_size):
                loss_classifier = classifier_criterion(pred_outputs[m_n][i].view([1, -1]), infer_input[i].view([-1]))
                classifier_optimizers[m_n].zero_grad()
                if i == (2 * batch_size) - 1:
                    loss_classifier.backward(retain_graph=False)
                else:
                    loss_classifier.backward(retain_graph=True)
                g = models[m_n].classifier.weight.grad.view([1, 1, 256, 100])

                if grads.size()[0] != 0:

                    grads = torch.cat((grads, g))

                else:
                    grads = g

            grads = torch.autograd.Variable(torch.from_numpy(grads.data.cpu().numpy()).cuda())
            c = torch.autograd.Variable(torch.from_numpy(correct.view([-1, 1]).data.cpu().numpy()).cuda())
            preds = torch.autograd.Variable(torch.from_numpy(pred_outputs[m_n].data.cpu().numpy()).cuda())
            models_outputs.append(preds)
            correct_labels.append(c)
            model_grads.append(grads)
        member_output = inference_model(model_grads, infer_input_one_hot, correct_labels, models_outputs)

        is_member_labels = torch.from_numpy(
            np.reshape(np.concatenate((np.zeros(v_mem_input.size(0)), np.ones(v_nonmem_input.size(0)))), [-1, 1])
        ).cuda()

        v_is_member_labels = torch.autograd.Variable(is_member_labels).type(torch.cuda.FloatTensor)

        loss = criterion(member_output, v_is_member_labels)

        # measure accuracy and record loss
        prec1 = np.mean((member_output.data.cpu().numpy() > 0.5) == v_is_member_labels.data.cpu().numpy())
        losses.update(loss.data, model_input.size(0))
        top1.update(prec1, model_input.size(0))

        # compute gradient and do SGD step

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if batch_idx % 10 == 0:
            pbar.write(
                "({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} ".format(
                    batch=batch_idx,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                )
            )

    return (losses.avg, top1.avg)


def mia_train(
    trainloader,
    testloader,
    model,
    inference_model,
    classifier_criterion,
    criterion,
    classifier_optimizer,
    optimizer,
    pbar,
    num_batchs=1000,
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    mtop1_a = AverageMeter()
    mtop5_a = AverageMeter()

    inference_model.train()
    model.eval()
    # switch to evaluate mode

    end = time.time()
    for batch_idx, ((mem_input, mem_target), (nonmem_input, nonmem_target)) in enumerate(zip(trainloader, testloader)):
        # measure data loading time
        if batch_idx > num_batchs:
            break
        data_time.update(time.time() - end)
        mem_input = mem_input.cuda()
        nonmem_input = nonmem_input.cuda()
        mem_target = mem_target.cuda()
        nonmem_target = nonmem_target.cuda()

        v_mem_input = torch.autograd.Variable(mem_input)
        v_nonmem_input = torch.autograd.Variable(nonmem_input)
        v_mem_target = torch.autograd.Variable(mem_target)
        v_nonmem_target = torch.autograd.Variable(nonmem_target)

        # compute output
        model_input = torch.cat((v_mem_input, v_nonmem_input))
        pred_outputs = model(model_input)

        infer_input = torch.cat((v_mem_target, v_nonmem_target))

        one_hot_tr = torch.from_numpy((np.zeros((infer_input.size(0), 100)))).cuda().type(torch.cuda.FloatTensor)
        target_one_hot_tr = one_hot_tr.scatter_(1, infer_input.type(torch.cuda.LongTensor).view([-1, 1]).data, 1)
        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)

        correct = torch.sum(pred_outputs * infer_input_one_hot, dim=1)
        grads = torch.zeros(0)

        for i in range(2 * batch_size):
            loss_classifier = classifier_criterion(pred_outputs[i].view([1, -1]), infer_input[i].view([-1]))
            classifier_optimizer.zero_grad()
            if i == (2 * batch_size) - 1:
                loss_classifier.backward(retain_graph=False)
            else:
                loss_classifier.backward(retain_graph=True)
            g = model.classifier.weight.grad.view([1, 1, 256, 100])

            if grads.size()[0] != 0:
                grads = torch.cat((grads, g))
            else:
                grads = g

        grads = torch.autograd.Variable(torch.from_numpy(grads.data.cpu().numpy()).cuda())
        c = torch.autograd.Variable(torch.from_numpy(correct.view([-1, 1]).data.cpu().numpy()).cuda())
        preds = torch.autograd.Variable(torch.from_numpy(pred_outputs.data.cpu().numpy()).cuda())

        member_output = inference_model(grads, infer_input_one_hot, c, preds)

        is_member_labels = torch.from_numpy(
            np.reshape(np.concatenate((np.zeros(v_mem_input.size(0)), np.ones(v_nonmem_input.size(0)))), [-1, 1])
        ).cuda()

        v_is_member_labels = torch.autograd.Variable(is_member_labels).type(torch.cuda.FloatTensor)

        loss = criterion(member_output, v_is_member_labels)

        # measure accuracy and record loss
        prec1 = np.mean((member_output.data.cpu().numpy() > 0.5) == v_is_member_labels.data.cpu().numpy())
        losses.update(loss.data, model_input.size(0))
        top1.update(prec1, model_input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if batch_idx % 10 == 0:
            pbar.write(
                "({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} ".format(
                    batch=batch_idx,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                )
            )

    return (losses.avg, top1.avg)


def mia_test(
    trainloader,
    testloader,
    model,
    inference_model,
    classifier_criterion,
    criterion,
    classifier_optimizer,
    pbar,
    num_batchs=1000,
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    mtop1_a = AverageMeter()
    mtop5_a = AverageMeter()

    inference_model.eval()
    model.eval()
    # switch to evaluate mode

    end = time.time()
    for batch_idx, ((mem_input, mem_target), (nonmem_input, nonmem_target)) in enumerate(zip(trainloader, testloader)):
        # measure data loading time
        if batch_idx > num_batchs:
            break
        data_time.update(time.time() - end)
        mem_input = mem_input.cuda()
        nonmem_input = nonmem_input.cuda()
        mem_target = mem_target.cuda()
        nonmem_target = nonmem_target.cuda()

        v_mem_input = torch.autograd.Variable(mem_input)
        v_nonmem_input = torch.autograd.Variable(nonmem_input)
        v_mem_target = torch.autograd.Variable(mem_target)
        v_nonmem_target = torch.autograd.Variable(nonmem_target)

        # compute output
        model_input = torch.cat((v_mem_input, v_nonmem_input))
        pred_outputs = model(model_input)

        infer_input = torch.cat((v_mem_target, v_nonmem_target))

        one_hot_tr = torch.from_numpy((np.zeros((infer_input.size(0), 100)))).cuda().type(torch.cuda.FloatTensor)
        target_one_hot_tr = one_hot_tr.scatter_(1, infer_input.type(torch.cuda.LongTensor).view([-1, 1]).data, 1)
        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)

        correct = torch.sum(pred_outputs * infer_input_one_hot, dim=1)
        grads = torch.zeros(0)

        for i in range(2 * batch_size):
            loss_classifier = classifier_criterion(pred_outputs[i].view([1, -1]), infer_input[i].view([-1]))
            classifier_optimizer.zero_grad()
            if i == (2 * batch_size) - 1:
                loss_classifier.backward(retain_graph=False)
            else:
                loss_classifier.backward(retain_graph=True)
            g = model.classifier.weight.grad.view([1, 1, 256, 100])

            if grads.size()[0] != 0:
                grads = torch.cat((grads, g))
            else:
                grads = g

        grads = torch.autograd.Variable(torch.from_numpy(grads.data.cpu().numpy()).cuda())
        c = torch.autograd.Variable(torch.from_numpy(correct.view([-1, 1]).data.cpu().numpy()).cuda())
        preds = torch.autograd.Variable(torch.from_numpy(pred_outputs.data.cpu().numpy()).cuda())

        member_output = inference_model(grads, infer_input_one_hot, c, preds)

        is_member_labels = torch.from_numpy(
            np.reshape(np.concatenate((np.zeros(v_mem_input.size(0)), np.ones(v_nonmem_input.size(0)))), [-1, 1])
        ).cuda()

        v_is_member_labels = torch.autograd.Variable(is_member_labels).type(torch.cuda.FloatTensor)

        loss = criterion(member_output, v_is_member_labels)

        # measure accuracy and record loss
        prec1 = np.mean((member_output.data.cpu().numpy() > 0.5) == v_is_member_labels.data.cpu().numpy())
        losses.update(loss.data, model_input.size(0))
        top1.update(prec1, model_input.size(0))

        # compute gradient and do SGD step

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if batch_idx % 10 == 0:
            pbar.write(
                "({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} ".format(
                    batch=batch_idx,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                )
            )

    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint="checkpoint", filename="checkpoint.pth.tar"):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "model_best.pth.tar"))


def adjust_learning_rate(optimizer, epoch, state):
    if epoch in [20, 40]:
        state["lr"] *= 0.1
        for param_group in optimizer.param_groups:
            param_group["lr"] = state["lr"]


def save_checkpoint_adversary(state, is_best, checkpoint="checkpoint", filename="checkpoint.pth.tar"):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "model_adversary_best.pth.tar"))


if __name__ == "__main__":
    mp.set_start_method("spawn")
    mp.set_sharing_strategy("file_system")

    dataset = "cifar100"
    checkpoint_path = "checkpoints_100cifar_alexnet_white_fed"
    train_batch = 100
    test_batch = 100
    lr = 0.05
    epochs = 100
    state = {}
    state["lr"] = lr
    cudnn.benchmark = True
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(checkpoint_path):
        mkdir_p(checkpoint_path)

    # Data
    print("==> Preparing dataset %s" % dataset)

    if dataset == "cifar10":
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    # Load dataset for membership inference
    title = "cifar-100"

    trainset = dataloader(root="./data100", train=True, download=True, transform=transforms.ToTensor())
    trainloader = data.DataLoader(trainset, batch_size=train_batch, shuffle=True)

    testset = dataloader(root="./data100", train=False, download=True, transform=transforms.ToTensor())
    testloader = data.DataLoader(testset, batch_size=test_batch, shuffle=True)

    r = np.arange(50000)
    np.random.shuffle(r)

    private_trainset_member = []
    private_trainset_nonmember = []

    private_testset_member = []
    private_testset_nonmember = []

    for i in range(25000):
        private_trainset_member.append(trainset[r[i]])

    for i in range(25000, 50000):
        private_testset_member.append(trainset[r[i]])

    r = np.arange(10000)
    np.random.shuffle(r)

    for i in range(5000):
        private_trainset_nonmember.append(testset[r[i]])

    for i in range(5000, 10000):
        private_testset_nonmember.append(testset[r[i]])

    batch_size = 20

    private_trainloader_member = data.DataLoader(private_trainset_member, batch_size=batch_size, shuffle=True)
    private_trainloader_nonmember = data.DataLoader(private_trainset_nonmember, batch_size=batch_size, shuffle=True)

    private_testloader_member = data.DataLoader(private_testset_member, batch_size=batch_size, shuffle=True)
    private_testloader_nonmember = data.DataLoader(private_testset_nonmember, batch_size=batch_size, shuffle=True)

    # load models for membership inference
    print("==> Loading models")

    criterion = nn.CrossEntropyLoss()

    net = AlexNet(num_classes)
    net = net.cuda()
    checkpoint = torch.load("checkpoints_100cifar_alexnet_white_fed/epoch_%d_main" % 299)
    net.load_state_dict(checkpoint["state_dict"])
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)

    # create attack model
    criterion_attack = nn.MSELoss()
    attack_mdoel = InferenceAttack_HZ(100)
    attack_mdoel = attack_mdoel.cuda()
    optimizer_attack = optim.Adam(attack_mdoel.parameters(), lr=0.0001)

    print("==> attack training")
    with tqdm(range(5)) as pbar:
        for epoch in pbar:
            mia_train(
                private_trainloader_member,
                private_trainloader_nonmember,
                net,
                attack_mdoel,
                criterion,
                criterion_attack,
                optimizer,
                optimizer_attack,
                pbar,
            )
            bb = mia_test(
                private_testloader_member,
                private_testloader_nonmember,
                net,
                attack_mdoel,
                criterion,
                criterion_attack,
                optimizer,
                pbar,
            )
            pbar.write(f"{bb}")

            save_checkpoint_adversary(
                {
                    "epoch": epoch + 1,
                    "state_dict": attack_mdoel.state_dict(),
                    "optimizer": optimizer_attack.state_dict(),
                },
                True,
                checkpoint=checkpoint_path,
            )
