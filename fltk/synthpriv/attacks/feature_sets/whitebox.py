from fltk.synthpriv.attacks.feature_sets.independent_histograms import HistogramFeatureSet
import numpy as np
import pandas as pd
import torch
from fltk.synthpriv.attacks.feature_sets.feature_set import FeatureSet
from fltk.synthpriv.attacks.feature_sets.model_agnostic import NaiveFeatureSet


class WhiteBoxFeatureSet(FeatureSet):
    """An ensemble of features that leverages pre-trained models"""

    def __init__(self, models, optimizers, criterion, num_classes, type="naive", num_features=300):
        self.models, self.optimizers, self.criterion, self.num_classes = models, optimizers, criterion, num_classes
        self.num_features = num_features
        if type == "naive":
            self.extractor = NaiveFeatureSet(np.ndarray)
        elif type == "hist":
            self.extractor = None  # delayed creation because metadata relies on whitebox data

    def extract(self, data):
        whitebox_data = self.whitebox(data)
        data = np.concatenate((data.values, whitebox_data), axis=1)

        if self.extractor is None:
            metadata = {"categorical_columns": [], "ordinal_columns": [], "continuous_columns": [], "columns": []}
            for n in range(data.shape[1]):
                unique_vals = np.unique(data[:, n])
                if len(unique_vals) <= 100:
                    metadata["categorical_columns"].append(n)
                    metadata["columns"].append(
                        {"name": n, "type": "categorical", "size": len(unique_vals), "i2s": unique_vals}
                    )
                else:
                    metadata["continuous_columns"].append(n)
                    metadata["columns"].append(
                        {"name": n, "type": "continuous", "min": data[:, n].min(), "max": data[:, n].max()}
                    )
            self.extractor = HistogramFeatureSet(np.ndarray, metadata=metadata)

        features = self.extractor.extract(data)
        return features

    def whitebox(self, data):
        self.models = [model.to("cuda", non_blocking=True) for model in self.models]

        labels = data["labels"]
        data = data.drop(columns=["labels"])

        predictions, correct, gradients = [], [], []
        for batch, lbls in zip(np.array_split(data.values, 128), np.array_split(labels.values, 128)):
            batch = torch.from_numpy(batch).cuda()
            lbls = torch.from_numpy(np.nan_to_num(lbls)).long().cuda()

            outputs = [
                model(
                    batch
                    if not model.__class__.__name__ in ["DenseNet", "AlexNet", "Cifar100ResNet"]
                    else batch.reshape(len(batch), 3, 32, 32)
                )
                for model in self.models
            ]

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

        self.models = [model.cpu() for model in self.models]
        torch.cuda.empty_cache()
        return np.concatenate(
            (
                np.concatenate(predictions, axis=0),
                np.concatenate(correct, axis=0),
                np.concatenate(gradients, axis=0),
            ),
            axis=1,
        )
