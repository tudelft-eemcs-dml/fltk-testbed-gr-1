import torch
from fltk.synthpriv.attacks.feature_sets.feature_set import FeatureSet
from fltk.synthpriv.attacks.feature_sets.model_agnostic import EnsembleFeatureSet
import numpy as np


class WhiteBoxFeatureSet(FeatureSet):
    """An ensemble of features that leverages the pre-trained model"""

    def __init__(self, models, optimizers, criterion, datatype, metadata, nbins=10):
        self.datatype = datatype
        self.ensemble = EnsembleFeatureSet(datatype, metadata, nbins)
        self.models, self.optimizers, self.criterion = models, optimizers, criterion

    def extract(self, data):
        F_ensemble = self.ensemble.extract(data)
        print(F_ensemble.shape)
        F_whitebox = self.whitebox(data)
        print(F_whitebox.shape)
        print(self.ensemble.extract(F_whitebox).shape)
        return np.concatenate([F_ensemble, F_whitebox])

    def whitebox(self, data):
        predictions, correct, gradients = [], [], []
        for batch in np.array_split(data, 20):
            inputs = torch.from_numpy(batch.drop(target))
            outputs = [model(inputs) for model in self.models]

            labels = torch.from_numpy((batch[target]))
            labels_1hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

            correct_labels = [np.sum(output * labels_1hot, dim=1).view([-1, 1]).cpu().numpy() for output in outputs]

            model_grads = []
            for m_n in range(len(self.models)):
                grads = []
                for i in range(len(batch)):
                    loss_classifier = self.criterion(outputs[m_n][i].view([1, -1]), labels[i].view([-1]))
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
