import numpy as np
import pandas as pd
import torch
from fltk.synthpriv.attacks.feature_sets.feature_set import FeatureSet
from fltk.synthpriv.attacks.feature_sets.model_agnostic import EnsembleFeatureSet


class WhiteBoxFeatureSet(FeatureSet):
    """An ensemble of features that leverages pre-trained models"""

    def __init__(self, metadata, models, optimizers, criterion, num_classes):
        self.models, self.optimizers, self.criterion, self.num_classes = models, optimizers, criterion, num_classes
        self.ensemble = EnsembleFeatureSet(pd.DataFrame, metadata, nbins=10)

    def extract(self, data):
        F_ensemble = self.ensemble.extract(data)
        print(F_ensemble.shape)

        F_whitebox = self.whitebox(data)
        print(F_whitebox.shape)

        return np.concatenate([F_ensemble, F_whitebox])

    def whitebox(self, data, labels):
        predictions, correct, gradients = [], [], []
        for batch, lbls in zip(np.array_split(data, 20), np.array_split(labels, 20)):
            outputs = [model(torch.from_numpy(batch).cuda()) for model in self.models]

            labels_1hot = torch.nn.functional.one_hot(lbls, num_classes=self.num_classes).float()

            correct_labels = [np.sum(output * labels_1hot, dim=1).view([-1, 1]).cpu().numpy() for output in outputs]

            model_grads = []
            for m_n in range(len(self.models)):
                grads = []
                for i in range(len(batch)):
                    loss_classifier = self.criterion(outputs[m_n][i].view([1, -1]), lbls[i].view([-1]))
                    self.optimizers[m_n].zero_grad()
                    loss_classifier.backward(retain_graph=i < len(batch) - 1)
                    g = list(self.models[m_n].parameters())[-2].grad.flatten()
                    grads.append(g.cpu().numpy())
                model_grads.append(np.concatenate(grads))

            predictions.append(np.concatenate(outputs, axis=1))
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
