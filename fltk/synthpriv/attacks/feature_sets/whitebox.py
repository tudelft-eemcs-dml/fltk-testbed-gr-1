import numpy as np
import pandas as pd
import torch
from fltk.synthpriv.attacks.feature_sets.feature_set import FeatureSet
from fltk.synthpriv.attacks.feature_sets.model_agnostic import NaiveFeatureSet


class WhiteBoxFeatureSet(FeatureSet):
    """An ensemble of features that leverages pre-trained models"""

    def __init__(self, metadata, models, optimizers, criterion, num_classes):
        self.models, self.optimizers, self.criterion, self.num_classes = models, optimizers, criterion, num_classes
        self.naive = NaiveFeatureSet(pd.DataFrame)

    def extract(self, data):
        whitebox_augmented_data = pd.DataFrame(np.concatenate((data.values, self.whitebox(data)), axis=1))
        return self.naive.extract(whitebox_augmented_data)

    def whitebox(self, data):
        labels = data["labels"]
        data = data.drop(columns=["labels"])

        predictions, correct, gradients = [], [], []
        self.models = [model.cuda() for model in self.models]
        for batch, lbls in zip(np.array_split(data.values, 20), np.array_split(labels.values, 20)):
            batch = torch.from_numpy(batch).cuda()
            lbls = torch.from_numpy(lbls.clip(0, self.num_classes)).long().cuda()

            outputs = [model(batch) for model in self.models]

            labels_1hot = torch.nn.functional.one_hot(lbls, num_classes=self.num_classes).float()

            model_grads, correct_labels = [], []
            for m_n in range(len(self.models)):
                grads = []
                for i in range(len(batch)):
                    loss_classifier = self.criterion(outputs[m_n][i].view([1, -1]), lbls[i].view([-1]))
                    self.optimizers[m_n].zero_grad()
                    loss_classifier.backward(retain_graph=i < len(batch) - 1)
                    g = list(self.models[m_n].parameters())[-2].grad.flatten()
                    grads.append(g.detach().cpu().numpy())
                model_grads.append(np.stack(grads, axis=0))
                correct_labels.append(torch.sum(outputs[m_n] * labels_1hot, dim=1).view([-1, 1]).detach().cpu().numpy())

            predictions.append(torch.cat(outputs, axis=1).detach().cpu().numpy())
            correct.append(np.concatenate(correct_labels, axis=1))
            gradients.append(np.concatenate(model_grads, axis=1))

        return np.concatenate(
            (
                np.concatenate(predictions, axis=0),
                np.concatenate(correct, axis=0),
                np.concatenate(gradients, axis=0),
            ),
            axis=1,
        )
