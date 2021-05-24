import logging
import os
import datetime

import joblib
import numpy as np
import torch
import matplotlib

from matplotlib import pyplot as plt
from torch import nn
from itertools import zip_longest
import torchextractor as tx
from sklearn.metrics import accuracy_score, auc, roc_curve



from fltk.synthpriv.attacks import NasrAttack, UnsupervisedNasrAttack
from fltk.synthpriv.config import SynthPrivConfig
from fltk.synthpriv.datasets.adult import DistAdultDataset
# from fltk.synthpriv.datasets.synthetic import SyntheticDataset
from fltk.synthpriv.models.adult_mlp import AdultMLP
from tqdm import tqdm


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


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        m.bias.data.fill_(0)


def fcn_module(inputsize, layer_size=128):
    """
    Creates a FCN submodule. Used in different attack components.
    """
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
    print("CNN 4 CNN")
    dim1, dim2, dim3, dim4 = input_shape
    cnn = nn.Sequential(
        [
            Print(),
            nn.Conv2d(dim4, dim4, kernel_size=(dim2, dim3), stride=(1, 1)),
            Print(),
            nn.ReLU(),
            nn.Flatten(),
            Print(),
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


def cnn_for_fcn_gradients(input_shape):
    """
    Creates a CNN submodule for Linear layer gradients.
    """
    dim1, dim2 = input_shape
    cnn = nn.Sequential(
        ReshapeForGradConv(),
        nn.Dropout(0.2),
        nn.Conv2d(1, 100, kernel_size=(1, dim2), stride=(1, 1)),
        nn.ReLU(),
        nn.Flatten(),
        nn.Dropout(0.2),
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
    print("CNN 4 CNN grads")
    dim2, dim3, dim4, dim1 = input_shape
    cnn = nn.Sequential(
        Print(),
        nn.Conv2d(dim1, kernel_size=(dim2, dim3), stride=(1, 1), padding="same", name="cnn_grad_layer"),
        Print(),
        nn.ReLU(),
        nn.Flatten(),
        nn.Dropout(0.2),
        Print(),
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

    target_model: The target classification model that'll be attacked

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
            target_model,
            train_dataloader,
            test_dataloader,
            layers_to_exploit=[],
            gradients_to_exploit=[],
            exploit_loss=True,
            exploit_label=True,
            optimizer=torch.optim.Adam,
            learning_rate=0.001,
            epochs=30,
    ):
        super().__init__()
        self.target_model = target_model.requires_grad_(False).eval()

        self.device = device
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.train_shape = next(iter(self.train_dataloader[0]))[0].shape

        self.layers_to_exploit = layers_to_exploit
        self.gradients_to_exploit = gradients_to_exploit
        self.exploit_loss = exploit_loss
        self.exploit_label = exploit_label

        self.n_labels = list(target_model.parameters())[-1].shape[0]

        self.create_attack_model()

        self.optimizer = optimizer(
            [p for n, p in self.named_parameters() if not "target_model" in n and not "feature_extractor" in n],
            lr=learning_rate,
        )
        self.epochs = epochs

        self.out_name = f"{self.__class__.__name__}_{self.target_model.__class__.__name__}_{datetime.datetime.now()}"

    def create_attack_model(self):
        self.input_modules = nn.ModuleList()
        classifier_input_size = 0

        if len(self.layers_to_exploit):
            layer_names_and_classes = [
                (n, m.__class__.__name__)
                for i, (n, m) in enumerate(self.target_model.named_modules())
                if i in self.layers_to_exploit
            ]
            self.layers_to_exploit, layer_classes = transpose(layer_names_and_classes)
            self.intermediate_feature_extractor = tx.Extractor(self.target_model, self.layers_to_exploit)

            example = next(iter(self.train_dataloader[0]))[0]
            layer_shapes = [v.shape[1] for v in self.intermediate_feature_extractor(example)[1].values()]

            for shape, type in zip(layer_shapes, layer_classes):
                requires_cnn = map(lambda i: i in type, CNN_COMPONENT_LIST)
                if any(requires_cnn):
                    module = cnn_for_cnn_layeroutputs(shape)
                else:
                    module = fcn_module(shape, 100)
                self.input_modules.append(module)
                classifier_input_size += 64

        if len(self.gradients_to_exploit):
            layers = list(self.target_model.modules())
            self.grad_exploit_layers = []
            for l in self.gradients_to_exploit:
                layer = layers[l]
                assert any(
                    map(lambda i: i in layer.__class__.__name__, GRAD_LAYERS_LIST)
                ), f"Only Linear & Conv layers are supported for gradient-based attacks"
                requires_cnn = map(lambda i: i in layer.__class__.__name__, CNN_COMPONENT_LIST)
                self.grad_exploit_layers.append(layer.weight)
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

    def compute_gradients(self, model, features, labels):
        gradients = []
        model.requires_grad_(True)
        logits = model(features)
        for l, label in enumerate(labels):
            loss = nn.functional.cross_entropy(logits[None, l], label[None])
            grads = torch.autograd.grad(loss, self.grad_exploit_layers, retain_graph=True)
            gradients.append(grads)
        model.requires_grad_(False)
        return [torch.stack(grads) for grads in transpose(gradients)]

    def get_gradient_norms(self, model, features, labels):
        return [torch.norm(grads[-1]) for grads in self.compute_gradients(model, features, labels)]

    def forward(self, model, features, labels):
        i = 0
        attack_input = []

        if len(self.gradients_to_exploit):
            model.requires_grad_(True)

        if len(self.layers_to_exploit):
            logits, intermediate_feature = self.intermediate_feature_extractor(features)
        else:
            logits = model(features)

        if len(self.layers_to_exploit):
            for layer_output in intermediate_feature.values():
                attack_input.append(self.input_modules[i](layer_output))
                i += 1

        individual_losses = []
        for l, label in enumerate(labels):
            individual_losses.append(nn.functional.cross_entropy(logits[None, l], label[None]))

        if len(self.gradients_to_exploit):
            gradients = [
                torch.autograd.grad(loss, self.grad_exploit_layers, retain_graph=True) for loss in individual_losses
            ]
            gradients = [torch.stack(grads) for grads in transpose(gradients)]
            for grads in gradients:
                attack_input.append(self.input_modules[i](grads))
                i += 1
            model.requires_grad_(False)

        if self.exploit_loss:
            loss = torch.tensor(individual_losses, device=self.device).mean()[None, None]
            loss_feature = self.input_modules[i](loss)
            loss_feature = torch.tile(loss_feature, (len(features), 1))
            attack_input.append(loss_feature)
            i += 1

        if self.exploit_label:
            preds = torch.argmax(logits, axis=1)
            preds = nn.functional.one_hot(preds, num_classes=self.n_labels).float()
            attack_input.append(self.input_modules[i](preds))
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
            mprobs = self.forward(self.target_model, mfeatures.to(self.device), mlabels.to(self.device))
            nonmprobs = self.forward(self.target_model, nmfeatures.to(self.device), nmlabels.to(self.device))
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
        self.to(self.device).train()

        mtestset, nmtestset = self.test_dataloader
        member_loader, nonmember_loader = self.train_dataloader

        nmfeat, nmlbl = transpose(nonmember_loader)
        preds = np.argmax(self.target_model(torch.cat(nmfeat).to(self.device)).cpu(), axis=1)
        acc = accuracy_score(np.concatenate(nmlbl), preds.cpu())
        print("Target model test accuracy", acc)

        best_accuracy = 0
        pbar = tqdm(range(self.epochs), desc="Training attack model...")
        for e in pbar:
            for (mfeatures, mlabels), (nmfeatures, nmlabels) in zip(member_loader, nonmember_loader):
                self.optimizer.zero_grad()
                moutputs = self.forward(self.target_model, mfeatures.to(self.device), mlabels.to(self.device))
                nmoutputs = self.forward(self.target_model, nmfeatures.to(self.device), nmlabels.to(self.device))

                memtrue = torch.ones(moutputs.shape, device=self.device)
                nonmemtrue = torch.zeros(nmoutputs.shape, device=self.device)

                target = torch.cat((memtrue, nonmemtrue))
                probs = torch.cat((moutputs, nmoutputs))

                attackloss = nn.functional.mse_loss(target, probs)
                attackloss.backward()
                self.optimizer.step()

            attack_accuracy = self.attack_accuracy(mtestset, nmtestset)
            if attack_accuracy > best_accuracy:
                best_accuracy = attack_accuracy
                best_state_dict = self.state_dict()

            pbar.write(f"Epoch {e} : Attack test accuracy: {attack_accuracy:.3f}, Best accuracy : {best_accuracy:.3f}")

        self.out_name += f"_{best_accuracy:.3f}"
        self.load_state_dict(best_state_dict)
        torch.save(
            self.cpu().eval().requires_grad_(False).state_dict(),
            f"models/{self.out_name}.pt",
        )

    def test_attack(self):
        """
        Test the attack model on dataset and save plots for visualization.
        """
        self.to(self.device).eval()

        mtrainset, nmtrainset = self.test_dataloader

        mpreds, mlab, nmpreds, nmlab, mfeat, nmfeat, mtrue, nmtrue = [], [], [], [], [], [], [], []
        mgradnorm, nmgradnorm = [], []

        for (mfeatures, mlabels) in mtrainset:
            moutputs = self.forward(self.target_model, mfeatures.to(self.device), mlabels.to(self.device)).detach()

            mpreds.extend(moutputs.cpu().numpy())
            mlab.extend(mlabels.cpu().numpy())
            mfeat.extend(mfeatures.cpu().numpy())
            mtrue.extend(np.ones(moutputs.shape))

            if len(self.gradients_to_exploit):
                mgradientnorm = self.get_gradient_norms(
                    self.target_model, mfeatures.to(self.device), mlabels.to(self.device)
                )
                mgradnorm.extend(mgradientnorm)

        for (nmfeatures, nmlabels) in nmtrainset:
            nmoutputs = self.forward(self.target_model, nmfeatures.to(self.device), nmlabels.to(self.device)).detach()

            nmpreds.extend(nmoutputs.cpu().numpy())
            nmlab.extend(nmlabels.cpu().numpy())
            nmfeat.extend(nmfeatures.cpu().numpy())
            nmtrue.extend(np.zeros(nmoutputs.shape))

            if len(self.gradients_to_exploit):
                nmgradientnorm = self.get_gradient_norms(
                    self.target_model, nmfeatures.to(self.device), nmlabels.to(self.device)
                )
                nmgradnorm.extend(nmgradientnorm)

        target = np.concatenate((np.concatenate(mtrue), np.concatenate(nmtrue)))
        probs = np.concatenate((np.concatenate(mpreds), np.concatenate(nmpreds)))

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
        if len(self.gradients_to_exploit):
            xs = []
            ys = []
            for lab in unique_mem_lab:
                gradnorm = []
                for l, p in zip(mlab, mgradientnorm):
                    if l == lab:
                        gradnorm.append(p.cpu().numpy())
                xs.append(lab)
                ys.append(np.mean(gradnorm))

            plt.plot(xs, ys, "g.", label="Training Data (Members)")

            xs = []
            ys = []
            for lab in unique_nmem_lab:
                gradnorm = []
                for l, p in zip(nmlab, nmgradientnorm):
                    if l == lab:
                        gradnorm.append(p.cpu().numpy())
                xs.append(lab)
                ys.append(np.mean(gradnorm))
            plt.plot(xs, ys, "r.", label="Population Data (Non-Members)")
            plt.title("Average Gradient Norms per Label")
            plt.xlabel("Label")
            plt.ylabel("Average Gradient Norm")
            plt.legend(loc="upper left")
            plt.savefig(f"output/{self.out_name}_gradient_norm.png")
            plt.close()


if __name__ == "__main__":
    import logging
    import sys

    implemented_datasets = ["purchase", "texas", "cifar-dense", "cifar-alex", "adult"]
    configs = {
        "purchase": {"data_location": "fltk/synthpriv/experiments/purchase.yaml"},
        "texas": {"data_location": "fltk/synthpriv/experiments/texas.yaml"},
        "cifar": {"data_location": "fltk/synthpriv/experiments/cifar100.yaml"}
    }
    if not len(sys.argv) == 3:
        print("Fill in the args please")

    selected_dataset = sys.argv[1]
    target_model_loc = sys.argv[2]

    if selected_dataset not in implemented_datasets:
        print("This dataset is not implemented yet, please choose from the following values:")
        print(implemented_datasets)

    import yaml
    from fltk.synthpriv.config import SynthPrivConfig

    from fltk.synthpriv.models.purchase_mlp import PurchaseMLP
    from fltk.synthpriv.datasets.purchase import DistPurchaseDataset

    from fltk.synthpriv.models.texas_mlp import TexasMLP
    from fltk.synthpriv.datasets.texas import DistTexasDataset

    from torchvision.models.densenet import densenet121 as Cifar100DenseNet
    from torchvision.models.alexnet import alexnet
    from fltk.synthpriv.datasets.cifar import DistCifarDataset

    from fltk.synthpriv.models.adult_mlp import AdultMLP
    from fltk.synthpriv.datasets.adult import DistAdultDataset

    from tqdm import tqdm

    torch.backends.cudnn.benchmark = True

    cfg = SynthPrivConfig()
    yaml_data = yaml.load(configs[selected_dataset]["data_location"], Loader=yaml.FullLoader)
    cfg.merge_yaml(yaml_data)
    cfg.init_logger(logging)

    print("loading target models")
    if selected_dataset == implemented_datasets[0]:
        target_model = PurchaseMLP()
        dataset = DistPurchaseDataset(cfg)
    elif selected_dataset == implemented_datasets[1]:
        target_model = TexasMLP()
        dataset = DistTexasDataset(cfg)
    elif selected_dataset == implemented_datasets[2]:
        target_model = Cifar100DenseNet()
        dataset = DistCifarDataset(cfg)
    elif selected_dataset == implemented_datasets[3]:
        target_model = alexnet()
        dataset = DistCifarDataset(cfg)
    elif selected_dataset == implemented_datasets[4]:
        target_model = AdultMLP()
        dataset = DistAdultDataset(cfg)

    target_model.load_state_dict(torch.load(target_model_loc))
    for i, (name, mod) in enumerate(target_model.named_modules()):
        print(i, name, mod.__class__.__name__)

    print("loading data")

    data_cache_path_root = f"data/nasr-attack-{dataset.__class__.__name__}"
    if not os.path.exists(data_cache_path_root + "-train.pkl"):
        member_loader = dataset.train_loader
        nonmember_loader = dataset.test_loader
        member_train, nonmember_train, member_test, nonmember_test = [], [], [], []
        pbar = tqdm(zip(member_loader, nonmember_loader), desc="Preparing data...", total=20_000)
        for (memfeat, memlabel), (nonmemfeat, nonmemlabel) in pbar:
            if len(member_test) * len(memfeat) < 15_000:
                member_test.append((memfeat, memlabel))
                nonmember_test.append((nonmemfeat, nonmemlabel))
            else:
                member_train.append((memfeat, memlabel))
                nonmember_train.append((nonmemfeat, nonmemlabel))
            if len(member_train) * len(memfeat) > 5_000:
                break
            pbar.update(len(memfeat))

        train_dataloader = member_train, nonmember_train
        test_dataloader = member_test, nonmember_test
        joblib.dump(train_dataloader, data_cache_path_root + "-train.pkl")
        joblib.dump(test_dataloader, data_cache_path_root + "-test.pkl")
    else:
        train_dataloader = joblib.load(data_cache_path_root + "-train.pkl")
        test_dataloader = joblib.load(data_cache_path_root + "-test.pkl")

    print("initalizing attack")
    attacker = NasrAttack(
        "cuda",
        target_model,
        train_dataloader,
        test_dataloader,
        layers_to_exploit=[10, 14],
        gradients_to_exploit=[10, 14],
        exploit_loss=True,
        exploit_label=True,
    )

    print("training attack model")
    attacker.train_attack()

    print("evaluating attack model")
    attacker.test_attack()
