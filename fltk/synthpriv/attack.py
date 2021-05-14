"""
Membership inference attack based on https://github.com/privacytrustlab/ml_privacy_meter/blob/master/ml_privacy_meter/attack/meminf.py
"""
import datetime
import os
from itertools import zip_longest

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchextractor as tx
from sklearn.metrics import accuracy_score, auc, roc_curve
from torch import nn


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        m.bias.data.fill_(0)


def fcn_module(inputsize, layer_size=128):
    """
    Creates a FCN submodule. Used in different attack components.
    """
    # print("FCN", inputsize)
    fcn = nn.Sequential(
        nn.Linear(inputsize, layer_size),
        nn.ReLU(),
        nn.Linear(layer_size, 64),
        nn.ReLU(),
    )
    fcn.apply(init_weights)
    return fcn


def cnn_for_cnn_layeroutputs(input_shape):
    """
    Creates a CNN submodule for Conv Layer outputs
    """
    # print("CNN^2", input_shape)
    dim1, dim2, dim3, dim4 = input_shape
    cnn = nn.Sequential(
        [
            nn.Conv2d(dim4, dim4, kernel_size=(dim2, dim3), stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(_, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        ]
    )
    cnn.apply(init_weights)
    return cnn


class ReshapeForGradConv(nn.Module):
    def forward(self, x):
        if x.dim() == 3:
            return x[:, None, ...]  # add channel dimension
        if x.dim() == 4:
            return x
        else:
            raise Exception("Only 3D and 4D inputs are supported to gradient convolution modules!")


class Print(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x


def cnn_for_fcn_gradients(input_shape):
    """
    Creates a CNN submodule for Linear layer gradients.
    """
    dim1, dim2 = input_shape
    cnn = nn.Sequential(
        # Print(),
        ReshapeForGradConv(),
        nn.Dropout(0.2),
        # Print(),
        nn.Conv2d(1, 100, kernel_size=(1, dim2), stride=(1, 1)),
        # Print(),
        nn.ReLU(),
        nn.Flatten(),
        nn.Dropout(0.2),
        # Print(),
        nn.Linear(100 * dim1, 2024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(2024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
    )
    cnn.apply(init_weights)
    return cnn


def cnn_for_cnn_gradients(input_shape):
    """
    Creates a CNN submodule for Conv layer gradients
    """
    # print("CNN^2 grad", input_shape)
    dim2, dim3, dim4, dim1 = input_shape
    cnn = nn.Sequential(
        nn.Conv2d(dim1, kernel_size=(dim2, dim3), stride=(1, 1), padding="same", name="cnn_grad_layer"),
        nn.ReLU(),
        nn.Flatten(name="flatten_layer"),
        nn.Dropout(0.2),
        nn.Linear(_, 64),
        nn.ReLU(),
    )
    cnn.apply(init_weights)
    return cnn


def transpose(l):
    return list(map(list, zip_longest(*l, fillvalue=None)))


# Decide what attack component (FCN or CNN) to use on the basis of the layer name.
# CNN_COMPONENTS_LIST are the layers requiring each input in 3 dimensions.
# GRAD_COMPONENTS_LIST are the layers which have trainable components for computing gradients
CNN_COMPONENT_LIST = ["Conv", "MaxPool"]
GRAD_LAYERS_LIST = ["Conv", "Linear"]


class NasrAttack(nn.Module):
    """
    This attack was originally proposed by Nasr et al. It exploits one-hot encoding of true labels, loss value,
    intermediate layer activations and gradients of intermediate layers of the target model on data points, for training
    the attack model to infer membership in training data.

    Paper link: https://arxiv.org/abs/1812.00910

    Args:
    ------
    device: torch.device() to use for training and testing

    target_train_model: The target (classification) model that'll be used to train the attack model

    target_test_model:  The target (classification) model to test privacy risk for

    train_dataloader: Dataloader with samples for training

    test_dataloader: Dataloader with samples for testing

    layers_to_exploit: a list of integers specifying the indices of layers, whose activations will be exploited by the
                       attack model. If the list has only a single element and it is equal to the index of last layer,
                       the attack can be considered as a "blackbox" attack.

    gradients_to_exploit: a list of integers specifying the indices of layers whose gradients will be exploited by the
                          attack model

    exploit_loss: boolean; whether to exploit loss value of target model or not

    exploit_label: boolean; whether to exploit one-hot encoded labels or not

    optimizer: The optimizer for training the attack model

    learning_rate: learning rate for training the attack model

    epochs: Number of epochs to train the attack model
    """

    def __init__(
        self,
        device,
        target_train_model,
        target_test_model,
        train_dataloader,
        test_dataloader,
        layers_to_exploit=None,
        gradients_to_exploit=None,
        exploit_loss=True,
        exploit_label=True,
        optimizer=torch.optim.Adam,
        learning_rate=0.001,
        epochs=100,
    ):
        super().__init__()

        self.device = device
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.layers_to_exploit = layers_to_exploit
        self.gradients_to_exploit = gradients_to_exploit
        self.exploit_loss = exploit_loss
        self.exploit_label = exploit_label

        self.n_labels = list(target_train_model.parameters())[-1].shape[0]

        self.create_attack_model(target_train_model)

        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.epochs = epochs

        self.target_train_model = target_train_model.requires_grad_(False).eval()
        self.target_test_model = target_test_model.requires_grad_(False).eval()

    def create_attack_model(self, target_train_model):
        self.input_modules = nn.ModuleList()
        classifier_input_size = 0

        if self.layers_to_exploit and len(self.layers_to_exploit):
            layer_names_and_classes = [
                (n, m.__class__.__name__)
                for i, (n, m) in enumerate(target_train_model.named_modules())
                if i in self.layers_to_exploit
            ]
            self.layers_to_exploit, layer_classes = transpose(layer_names_and_classes)
            example = next(iter(self.train_dataloader[0]))[0]
            layer_shapes = [
                v.shape[1] for v in tx.Extractor(target_train_model, self.layers_to_exploit)(example)[1].values()
            ]
            for shape, type in zip(layer_shapes, layer_classes):
                requires_cnn = map(lambda i: i in type, CNN_COMPONENT_LIST)
                if any(requires_cnn):
                    module = cnn_for_cnn_layeroutputs(shape)
                else:
                    module = fcn_module(shape, 100)
                self.input_modules.append(module)
                classifier_input_size += 64

        if self.gradients_to_exploit and len(self.gradients_to_exploit):
            layers = list(target_train_model.modules())
            for g, l in enumerate(self.gradients_to_exploit):
                layer = layers[l]
                assert any(
                    map(lambda i: i in layer.__class__.__name__, GRAD_LAYERS_LIST)
                ), f"Only Linear & Conv layers are supported for gradient-based attacks"
                requires_cnn = map(lambda i: i in layer.__class__.__name__, CNN_COMPONENT_LIST)
                self.gradients_to_exploit[g] = layer.weight
                if any(requires_cnn):
                    module = cnn_for_cnn_gradients(layer.weight.shape)
                    classifier_input_size += 64
                else:
                    module = cnn_for_fcn_gradients(layer.weight.shape)
                    classifier_input_size += 256
                self.input_modules.append(module)

        if self.exploit_loss:
            self.input_modules.append(fcn_module(1, 100))
            classifier_input_size += 64

        if self.exploit_label:
            self.input_modules.append(fcn_module(self.n_labels))
            classifier_input_size += 64

        classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        classifier.apply(init_weights)
        self.classifier = classifier
        print(self)

    def get_layer_outputs(self, model, features):
        extractor = tx.Extractor(model, self.layers_to_exploit)
        _, intermediate = extractor(features)
        return list(intermediate.values())

    def get_labels(self, labels):
        return nn.functional.one_hot(labels, num_classes=self.n_labels).float()

    def get_loss(self, model, features, labels):
        return nn.functional.cross_entropy(model(features), labels)

    def compute_gradients(self, model, features, labels):
        gradient_arr = []
        model.requires_grad_(True)
        for feature, label in zip(features, labels):
            loss = nn.functional.cross_entropy(model(feature[None, :]), label[None])
            grads = torch.autograd.grad(loss, self.gradients_to_exploit)
            gradient_arr.append(grads)
        model.requires_grad_(False)
        return gradient_arr

    def get_gradients(self, model, features, labels):
        gradient_arr = self.compute_gradients(model, features, labels)
        return [torch.stack(grads) for grads in transpose(gradient_arr)]

    def get_gradient_norms(self, model, features, labels):
        gradient_arr = self.compute_gradients(model, features, labels)
        return [torch.norm(grads[-1]) for grads in gradient_arr]

    def forward(self, model, features, labels):
        i = 0
        attack_input = []

        if self.layers_to_exploit and len(self.layers_to_exploit):
            for layer_output in self.get_layer_outputs(model, features):
                attack_input.append(self.input_modules[i](layer_output))
                i += 1

        if self.gradients_to_exploit and len(self.gradients_to_exploit):
            for layer_gradients in self.get_gradients(model, features, labels):
                attack_input.append(self.input_modules[i](layer_gradients))
                i += 1

        if self.exploit_loss:
            loss = self.get_loss(model, features, labels)[None, None]
            loss_feature = self.input_modules[i](loss)
            loss_feature = torch.tile(loss_feature, (len(features), 1))
            attack_input.append(loss_feature)
            i += 1

        if self.exploit_label:
            attack_input.append(self.input_modules[i](self.get_labels(labels)))
            i += 1

        return self.classifier(torch.cat(attack_input, axis=1))

    def attack_accuracy(self, members, nonmembers):
        """
        Computes attack accuracy of the attack model.
        """
        preds, targets = [], []
        for (membatch, nonmembatch) in zip(members, nonmembers):
            mfeatures, mlabels = membatch
            nmfeatures, nmlabels = nonmembatch

            # Computing the membership probabilities
            mprobs = self.forward(self.target_train_model, mfeatures.to(self.device), mlabels.to(self.device))
            nonmprobs = self.forward(self.target_train_model, nmfeatures.to(self.device), nmlabels.to(self.device))
            probs = torch.cat((mprobs, nonmprobs)).cpu()

            target_ones = torch.ones(mprobs.shape, dtype=bool)
            target_zeros = torch.zeros(nonmprobs.shape, dtype=bool)
            target = torch.cat((target_ones, target_zeros))

            preds.append(probs > 0.5)
            targets.append(target)

        return accuracy_score(np.concatenate(preds), np.concatenate(targets))

    def train_attack(self):
        """
        Trains the attack model
        """
        assert self.classifier, "Attack model not initialized"
        best_state_dict = self.state_dict()
        self.to(self.device)
        self.target_train_model.to(self.device)

        mtestset, nmtestset = self.test_dataloader
        member_loader, nonmember_loader = self.train_dataloader

        nmfeat, nmlbl = transpose(nonmember_loader)
        preds = np.argmax(self.target_train_model(torch.cat(nmfeat).to(self.device)).cpu(), axis=1)
        acc = accuracy_score(np.concatenate(nmlbl), preds.cpu())
        print("Target model test accuracy", acc)

        best_accuracy = 0
        pbar = tqdm(range(self.epochs), desc="Training attack model...")
        for e in pbar:
            for (mfeatures, mlabels), (nmfeatures, nmlabels) in zip(member_loader, nonmember_loader):
                self.optimizer.zero_grad()
                moutputs = self.forward(self.target_train_model, mfeatures.to(self.device), mlabels.to(self.device))
                nmoutputs = self.forward(self.target_train_model, nmfeatures.to(self.device), nmlabels.to(self.device))

                memtrue = torch.ones(moutputs.shape, device=self.device)
                nonmemtrue = torch.zeros(nmoutputs.shape, device=self.device)

                target = torch.cat((memtrue, nonmemtrue))
                probs = torch.cat((moutputs, nmoutputs))

                attackloss = nn.functional.mse_loss(target, probs)
                attackloss.backward()
                self.optimizer.step()

            pbar.write(f"{target.detach().squeeze().cpu().numpy()}")
            pbar.write(f"{probs.detach().squeeze().cpu().numpy()}")

            attack_accuracy = self.attack_accuracy(mtestset, nmtestset)
            if attack_accuracy > best_accuracy:
                best_accuracy = attack_accuracy
                best_state_dict = self.state_dict()

            pbar.write(f"Epoch {e} : Attack test accuracy: {attack_accuracy:.3f}, Best accuracy : {best_accuracy:.3f}")

        self.out_name = f"{self.__class__.__name__}_{self.target_train_model.__class__.__name__}_{best_accuracy:.3f}_{datetime.datetime.now()}"
        output_model = self
        output_model.load_state_dict(best_state_dict)
        torch.save(
            output_model.cpu().eval().requires_grad_(False).state_dict(),
            f"models/{self.out_name}.pt",
        )

    def test_attack(self):
        """
        Test the attack model on dataset and save plots for visualization.
        """
        mtrainset, nmtrainset = self.test_dataloader

        mpreds, mlab, nmpreds, nmlab, mfeat, nmfeat, mtrue, nmtrue = [], [], [], [], [], [], [], []
        mgradnorm, nmgradnorm = [], []

        for (mfeatures, mlabels) in mtrainset:
            moutputs = self.forward(self.target_test_model, mfeatures, mlabels)
            mgradientnorm = self.get_gradient_norms(self.target_test_model, mfeatures, mlabels)

            mpreds.extend(moutputs.numpy())
            mlab.extend(mlabels)
            mfeat.extend(mfeatures)
            mgradnorm.extend(mgradientnorm)
            mtrue.extend(np.ones(moutputs.shape))

        for (nmfeatures, nmlabels) in nmtrainset:
            nmoutputs = self.forward(self.target_test_model, nmfeatures, nmlabels)
            nmgradientnorm = self.get_gradient_norms(self.target_test_model, nmfeatures, nmlabels)

            nmpreds.extend(nmoutputs.numpy())
            nmlab.extend(nmlabels)
            nmfeat.extend(nmfeatures)
            nmgradnorm.extend(nmgradientnorm)
            nmtrue.extend(np.zeros(nmoutputs.shape))

        target = torch.cat((mtrue, nmtrue))
        probs = torch.cat((mpreds, nmpreds))

        font = {"weight": "bold", "size": 10}

        matplotlib.rc("font", **font)
        unique_mem_lab = sorted(np.unique(mlab))
        unique_nmem_lab = sorted(np.unique(nmlab))

        # Creates a histogram for Membership Probability
        fig = plt.figure(1)
        plt.hist(
            np.array(mpreds).flatten(),
            color="xkcd:blue",
            alpha=0.7,
            bins=20,
            histtype="bar",
            range=(0, 1),
            weights=(np.ones_like(mpreds) / len(mpreds)),
            label="Training Data (Members)",
        )
        plt.hist(
            np.array(nmpreds).flatten(),
            color="xkcd:light blue",
            alpha=0.7,
            bins=20,
            histtype="bar",
            range=(0, 1),
            weights=(np.ones_like(nmpreds) / len(nmpreds)),
            label="Population Data (Non-members)",
        )
        plt.xlabel("Membership Probability")
        plt.ylabel("Fraction")
        plt.title("Privacy Risk")
        plt.legend(loc="upper left")
        plt.savefig(f"output/{self.out_name}_privacy_risk.png")
        plt.close()

        # Creates ROC curve for membership inference attack
        fpr, tpr, _ = roc_curve(target, probs)
        roc_auc = auc(fpr, tpr)
        plt.title("ROC of Membership Inference Attack")
        plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.savefig(f"output/{self.out_name}_roc.png")
        plt.close()

        # Creates plot for gradient norm per label
        xs = []
        ys = []
        for lab in unique_mem_lab:
            gradnorm = []
            for l, p in zip(mlab, mgradientnorm):
                if l == lab:
                    gradnorm.append(p)
            xs.append(lab)
            ys.append(np.mean(gradnorm))

        plt.plot(xs, ys, "g.", label="Training Data (Members)")

        xs = []
        ys = []
        for lab in unique_nmem_lab:
            gradnorm = []
            for l, p in zip(nmlab, nmgradientnorm):
                if l == lab:
                    gradnorm.append(p)
            xs.append(lab)
            ys.append(np.mean(gradnorm))
        plt.plot(xs, ys, "r.", label="Population Data (Non-Members)")
        plt.title("Average Gradient Norms per Label")
        plt.xlabel("Label")
        plt.ylabel("Average Gradient Norm")
        plt.legend(loc="upper left")
        plt.savefig(f"output/{self.out_name}_gradient_norm.png")
        plt.close()

        # Collect data and creates histogram for each label separately
        for lab in range(len(unique_mem_lab)):
            labs = []
            for l, p in zip(mlab, mpreds):
                if l == lab:
                    labs.append(p)

            plt.hist(
                np.array(labs).flatten(),
                color="xkcd:blue",
                alpha=0.7,
                bins=20,
                label="Training Data (Members)",
                histtype="bar",
                range=(0, 1),
                weights=(np.ones_like(labs) / len(labs)),
            )

            labs = []
            for l, p in zip(nmlab, nmpreds):
                if l == lab:
                    labs.append(p)

            plt.hist(
                np.array(labs).flatten(),
                color="xkcd:light blue",
                alpha=0.7,
                bins=20,
                label="Population Data (Non-members)",
                histtype="bar",
                range=(0, 1),
                weights=(np.ones_like(labs) / len(labs)),
            )

            plt.legend()
            plt.xlabel("Membership Probability")
            plt.ylabel("Fraction")

            plt.title("Privacy Risk - Class " + str(lab))
            plt.savefig(f"output/{self.out_name}_privacy_risk_label" + str(lab) + ".png")

            plt.close()


if __name__ == "__main__":
    import logging

    import yaml
    from fltk.synthpriv.config import SynthPrivConfig
    from fltk.synthpriv.datasets.purchase import DistPurchaseDataset
    from fltk.synthpriv.models.purchase_mlp import PurchaseMLP

    from tqdm import tqdm

    torch.backends.cudnn.benchmark = True

    cfg = SynthPrivConfig()
    yaml_data = yaml.load("fltk/synthpriv/experiments/purchase.yaml", Loader=yaml.FullLoader)
    cfg.merge_yaml(yaml_data)
    cfg.init_logger(logging)

    print("loading target models")
    train_model = PurchaseMLP()
    train_model.load_state_dict(torch.load("models/model_client1_27_end.model"))
    test_model = PurchaseMLP()
    test_model.load_state_dict(torch.load("models/model_client1_100_end.model"))
    for i, (name, mod) in enumerate(train_model.named_modules()):
        print(i, name, mod.__class__.__name__)

    print("loading data")
    if not os.path.exists("data/nasr-attack-train.pkl"):
        dataset = DistPurchaseDataset(cfg)
        member_loader = dataset.train_loader
        nonmember_loader = dataset.test_loader

        test_frac = 0.25
        num_batches = len(member_loader)
        member_train, nonmember_train, member_test, nonmember_test = [], [], [], []
        for i, (features, labels) in enumerate(tqdm(member_loader)):
            if i < test_frac * num_batches:
                member_test.append((features, labels))
            else:
                member_train.append((features, labels))
        num_batches = len(nonmember_loader)
        for i, (features, labels) in enumerate(tqdm(nonmember_loader)):
            if i < test_frac * num_batches:
                nonmember_test.append((features, labels))
            else:
                nonmember_train.append((features, labels))

        train_dataloader = member_train, nonmember_train
        test_dataloader = member_test, nonmember_test

        joblib.dump(train_dataloader, "data/nasr-attack-train.pkl")
        joblib.dump(test_dataloader, "data/nasr-attack-test.pkl")
    else:
        train_dataloader = joblib.load("data/nasr-attack-train.pkl")
        test_dataloader = joblib.load("data/nasr-attack-test.pkl")

    print("initalizing attack")
    attacker = NasrAttack(
        "cuda",
        train_model,
        test_model,
        train_dataloader,
        test_dataloader,
        layers_to_exploit=[15],
        gradients_to_exploit=[14],
        exploit_loss=True,
        exploit_label=True,
    )

    print("training attack model")
    attacker.train_attack()

    print("evaluating attack model")
    attacker.test_attack()
