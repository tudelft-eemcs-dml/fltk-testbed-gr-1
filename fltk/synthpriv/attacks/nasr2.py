import torch
import torch.nn as nn
import torch.nn.parallel
from tqdm import tqdm


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


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        m.bias.data.fill_(0)


class NasrAttackV2(nn.Module):
    def __init__(self, num_classes, gradient_shape, num_models=1):
        super(NasrAttackV2, self).__init__()

        self.num_classes = num_classes
        self.num_models = num_models
        assert len(gradient_shape) == 2, "Only layers with 2D gradients are supported"

        self.grad_out, self.grad_in = gradient_shape

        grad_conv_channels = 512
        self.grads = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv2d(1, grad_conv_channels, kernel_size=(1, self.grad_out), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(self.grad_in * grad_conv_channels, 1024),
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
        self.apply(init_weights)

    def forward(self, gs, l, cs, os):
        for i in range(self.num_models):
            out_g = self.grads(gs[i])
            out_l = self.labels(l)  # labels are the same for all target models
            out_c = self.correct(cs[i])
            out_o = self.preds(os[i])
            if i == 0:
                com_inp = torch.cat((out_g, out_l, out_c, out_o), 1)
            else:
                com_inp = torch.cat((out_g, out_l, out_c, out_o, com_inp), 1)
        is_member = self.combine(com_inp)
        return self.output(is_member)


def attack(
    attack_model,
    memloader,
    nonmemloader,
    criterion,
    optimizer,
    classifiers,
    classifier_criterion,
    classifier_optimizers,
    num_batches=10,
):
    losses, top1 = AverageMeter(), AverageMeter()

    attack_model.train() if optimizer else attack_model.eval()
    classifiers = [model.eval().cuda() for model in classifiers]

    data_iter = iter(zip(memloader, nonmemloader))

    for _ in range(num_batches):
        (mem_input, mem_target), (nonmem_input, nonmem_target) = next(data_iter)
        mem_input, nonmem_input = mem_input.cuda(), nonmem_input.cuda()
        mem_target, nonmem_target = mem_target.cuda(), nonmem_target.cuda()

        model_input = torch.cat((mem_input, nonmem_input))
        pred_outputs = [classifier(model_input) for classifier in classifiers]

        labels = torch.cat((mem_target, nonmem_target))
        labels_1hot = nn.functional.one_hot(labels, num_classes=attack_model.num_classes).float()

        classifiers_outputs, correct_labels, model_grads = [], [], []
        for m_n in range(len(classifiers)):
            correct = torch.sum(pred_outputs[m_n] * labels_1hot, dim=1)
            grads = []

            for i in range(len(model_input)):
                loss_classifier = classifier_criterion(pred_outputs[m_n][i].view([1, -1]), labels[i].view([-1]))
                classifier_optimizers[m_n].zero_grad()
                loss_classifier.backward(retain_graph=i < len(model_input) - 1)
                g = list(classifiers[m_n].parameters())[-2].grad.view(
                    [1, 1, attack_model.grad_in, attack_model.grad_out]
                )
                grads.append(g)

            classifiers_outputs.append(pred_outputs[m_n].detach())
            correct_labels.append(correct.view([-1, 1]).detach())
            model_grads.append(torch.cat(grads).detach())

        attack_output = attack_model(model_grads, labels_1hot, correct_labels, classifiers_outputs)
        is_member = torch.cat(
            (torch.ones(len(mem_input), device="cuda"), torch.zeros(len(nonmem_input), device="cuda"))
        )[:, None]

        if optimizer:
            loss = criterion(attack_output, is_member)
            losses.update(loss.detach().item(), model_input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        prec1 = torch.mean(((attack_output > 0.5) == is_member).float()).detach().item()
        top1.update(prec1, model_input.size(0))

    return losses.avg, top1.avg