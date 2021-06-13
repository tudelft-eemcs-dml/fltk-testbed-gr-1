""" Membership inference attack on synthetic data that implements the risk of linkability. """

import json
import multiprocessing
import os
from argparse import ArgumentParser
from multiprocessing import Pool
from os import path
from time import time

import pandas as pd
from fltk.synthpriv.plot import plot_hist, plot_roc
from numpy import arange, array, concatenate, mean, ndarray, round, stack
from numpy.random import choice
from pandas import DataFrame
from pandas.api.types import CategoricalDtype, is_numeric_dtype
from sklearn.impute import SimpleImputer
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm

from .feature_sets.bayes import CorrelationsFeatureSet
from .feature_sets.independent_histograms import HistogramFeatureSet
from .feature_sets.model_agnostic import EnsembleFeatureSet, NaiveFeatureSet

LABEL_OUT = 0
LABEL_IN = 1

FILLNA_VALUE_CAT = "NaN"
CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
ORDINAL = "ordinal"


def get_auc(trueLables, scores):
    fpr, tpr, _ = roc_curve(trueLables, scores)
    area = auc(fpr, tpr)
    return area


def get_scores(classProbabilities, trueLabels):
    return [p[l] for p, l in zip(classProbabilities, trueLabels)]


def get_fp_tn_fn_tp(guesses, trueLabels):
    fp = sum([g == LABEL_IN for g, s in zip(guesses, trueLabels) if s == LABEL_OUT])
    tn = sum([g == LABEL_OUT for g, s in zip(guesses, trueLabels) if s == LABEL_OUT])
    fn = sum([g == LABEL_OUT for g, s in zip(guesses, trueLabels) if s == LABEL_IN])
    tp = sum([g == LABEL_IN for g, s in zip(guesses, trueLabels) if s == LABEL_IN])
    return fp, tn, fn, tp


def get_attack_accuracy(tp, tn, nguesses):
    return (tp + tn) / nguesses


def get_attack_precision(fp, tp):
    try:
        return tp / (tp + fp)
    except ZeroDivisionError as e:
        return 0.5


def get_attack_recall(tp, fn):
    try:
        return tp / (tp + fn)
    except ZeroDivisionError as e:
        return 0.5


def get_record_privacy_loss(pSuccess, pPrior):
    return pSuccess - pPrior


def get_record_privacy_gain(privLossRawT, privLossSynT):
    return (privLossRawT - privLossSynT) / 2


class MIAttackClassifier:
    """ " Parent class for membership inference attacks based on shadow modelling
    using a sklearn classifier as attack model"""

    def __init__(self, AttackClassifier, metadata, priorProbabilities, FeatureSet=None):
        """

        :param AttackClassifier: Classifier: An object that implements a binary classifier
        :param metadata: dict: Attribute metadata describing the data domain of the synthetic target data
        :param priorProbabilities: dict: Prior probabilities over the target's membership
        :param FeatureSet: FeatureSet: An object that implements a feacture extraction strategy for converting a dataset into a feature vector
        """

        self.AttackClassifier = AttackClassifier
        self.FeatureSet = FeatureSet
        self.ImputerCat = SimpleImputer(strategy="most_frequent")
        self.ImputerNum = SimpleImputer(strategy="median")
        self.metadata = metadata

        self.priorProbabilities = priorProbabilities

        self.trained = False

        self.__name__ = f"{self.AttackClassifier.__class__.__name__}{self.FeatureSet.__class__.__name__}"

    def _get_prior_probability(self, secret):
        """Get prior probability of the adversary guessing the target's secret

        :param secret: int: Target's true secret. Either LABEL_IN=1 or LABEL_OUT=0
        """
        try:
            return self.priorProbabilities[secret]
        except:
            return 0

    def train(self, synA, labels):
        """Train a membership inference attack on a labelled training set

        :param synA: list of ndarrays: A list of synthetic datasets
        :param labels: list: A list of labels that indicate whether target was in the training data (LABEL_IN=1) or not (LABEL_OUT=0)
        """

        if self.FeatureSet is not None:
            synA = stack([self.FeatureSet.extract(s) for s in synA])
        else:
            synA = [self._impute_missing_values(s) for s in synA]
            synA = stack([convert_df_to_array(s, self.metadata).flatten() for s in synA])
        if not isinstance(labels, ndarray):
            labels = array(labels)

        self.AttackClassifier.fit(synA, labels)
        self.trained = True

        del synA, labels

    def attack(self, testData):
        """Makes a guess about target's presence in the training set of the model that produced the synthetic input data

        :param testData: ndarray or DataFrame: A synthetic dataset
        """

        assert self.trained, "Attack must first be trained before can predict membership"

        if self.FeatureSet is not None:
            testData = stack([self.FeatureSet.extract(s) for s in testData])
        else:
            testData = stack([convert_df_to_array(s, self.metadata).flatten() for s in testData])

        rounded_prediction = self.AttackClassifier.predict(testData)
        prediction_prob = self.AttackClassifier.predict_proba(testData)
        return [prediction_prob.astype(float).tolist(), round(rounded_prediction, 0).astype(int).tolist()]

    def get_probability_of_success(self, testData, secret):
        """Calculate probability that attacker correctly predicts whether target was present in model's training data

        :param testData: ndarray or DataFrame: A synthetic dataset
        :param secret: int: Target's true secret. Either LABEL_IN=1 or LABEL_OUT=0
        """
        assert (
            self.trained
        ), "Attack must first be trained on some random data before can predict membership of target data"

        if self.FeatureSet is not None:
            testData = stack([self.FeatureSet.extract(s) for s in testData])
        else:
            testData = stack([convert_df_to_array(s, self.metadata).flatten() for s in testData])

        probs = self.AttackClassifier.predict_proba(testData)
        return [p[s] for p, s in zip(probs, secret)]

    def _impute_missing_values(self, df):
        """Impute missing values in a DataFrame

        :param df: DataFrame
        """

        cat_cols = list(df.select_dtypes(["object", "category"]))
        if len(cat_cols) > 0:
            self.ImputerCat.fit(df[cat_cols])
            df[cat_cols] = self.ImputerCat.transform(df[cat_cols])

        num_cols = list(df.select_dtypes(["int", "float"]))
        if len(num_cols) > 0:
            self.ImputerNum.fit(df[num_cols])
            df[num_cols] = self.ImputerNum.transform(df[num_cols])

        return df


def load_local_data_as_df(filename):
    with open(f"{filename}.json") as f:
        metadata = json.load(f)
    dtypes = {cd["name"]: (float if cd["type"] == "continuous" else str) for cd in metadata["columns"]}
    df = pd.read_csv(f"{filename}.csv", dtype=dtypes)
    metadata["categorical_columns"] = list()
    metadata["ordinal_columns"] = list()
    metadata["continuous_columns"] = list()
    for column_idx, column in enumerate(metadata["columns"]):
        if column["type"] == CATEGORICAL:
            metadata["categorical_columns"].append(column_idx)
        elif column["type"] == ORDINAL:
            metadata["ordinal_columns"].append(column_idx)
        elif column["type"] == CONTINUOUS:
            metadata["continuous_columns"].append(column_idx)
    return df, metadata


def convert_df_to_array(df, metadata):
    dfcopy = df.copy()
    for col in metadata["columns"]:
        if col["name"] in list(dfcopy):
            col_data = dfcopy[col["name"]]
            if col["type"] in [CATEGORICAL, ORDINAL]:
                if len(col_data) > len(col_data.dropna()):
                    col_data = col_data.fillna(FILLNA_VALUE_CAT)
                    if FILLNA_VALUE_CAT not in col["i2s"]:
                        col["i2s"].append(FILLNA_VALUE_CAT)
                        col["size"] += 1
                cat = CategoricalDtype(categories=col["i2s"], ordered=True)
                col_data = col_data.astype(cat)
                dfcopy[col["name"]] = col_data.cat.codes
    return dfcopy.values


def craft_outlier(data, size):
    # Craft outlier target
    outlier = DataFrame(columns=list(data))
    for attr in list(data):
        if is_numeric_dtype(data[attr]):
            outlier[attr] = [data[attr].max() for _ in range(size)]
        else:
            outlier[attr] = [data[attr].value_counts().index[-1] for _ in range(size)]
    outlier.index = ["Crafted" for _ in range(size)]
    return outlier


def generate_mia_shadow_data_shufflesplit(target, attackData, sizeRaw, numModels):
    """Procedure to train a set of shadow models on multiple training sets sampled from a reference dataset.

    :param target: ndarray or DataFrame: The target record
    :param attackData: ndarray or DataFrame: Attacker's reference dataset of size n_A
    :param sizeRaw: int: Size of the target training set
    :param numModels: int: Number of shadow models to train

    :returns
        :return synA: list of ndarrays or DataFrames: List of synthetic datasets
        :return labels: list: List of labels indicating whether target was in or out
    """
    kf = ShuffleSplit(n_splits=numModels, train_size=sizeRaw)
    synA, labels = [], []

    for train_index, _ in kf.split(attackData):
        # Sample a new training set from the reference dataset
        if isinstance(attackData, DataFrame):
            attackDataOut = attackData.iloc[train_index]
        else:
            attackDataOut = attackData[train_index, :]
        synA.extend([attackDataOut])
        labels.extend([LABEL_OUT])

        # Insert target record into training data
        if isinstance(attackData, DataFrame):
            attackDataIn = attackDataOut.append(target)
        else:
            if len(target.shape) == 1:
                target = target.reshape(1, len(target))
            attackDataIn = concatenate([attackDataOut, target])
        synA.extend([attackDataIn])
        labels.extend([LABEL_IN])

    return synA, labels


def generate_mia_shadow_data_allin(target, attackData):
    """Generate training data for the MIA from a *single* shadow model trained on the entire reference dataset at once

    :param target: ndarray or DataFrame: The target record
    :param attackData: ndarray or DataFrame: Attacker's reference dataset of size n_A

    :returns
        :return synA: list of ndarrays or DataFrames: List of synthetic datasets
        :return labels: list: List of labels indicating whether target was in or out
    """
    synA, labels = [], []

    # Generate synthetic sample for data without target
    synA.extend([attackData])
    labels.extend([LABEL_OUT])

    # Insert targets into training data
    if isinstance(attackData, DataFrame):
        attackDataIn = attackData.append(target)
    else:
        if len(target.shape) == 1:
            target = target.reshape(1, len(target))
        attackDataIn = concatenate([attackData, target])
    synA.extend([attackDataIn])
    labels.extend([LABEL_IN])

    return synA, labels


def run_mia(Attack, testData, testLabels, targetID, nr):
    probSuccessSyn = Attack.get_probability_of_success(testData, testLabels)
    priorProb = [Attack._get_prior_probability(s) for s in testLabels]
    privLossSyn = [get_record_privacy_loss(ps, pp) for ps, pp in zip(probSuccessSyn, priorProb)]
    privLossRaw = [get_record_privacy_loss(1, pp) for pp in priorProb]
    privGain = [get_record_privacy_gain(plR, plS) for plR, plS in zip(privLossRaw, privLossSyn)]

    predictions = Attack.attack(testData)
    rounded_prediction = predictions[1]
    predictions = predictions[0]
    fp, tn, fn, tp = get_fp_tn_fn_tp(rounded_prediction, testLabels)

    results = {
        "TargetID": [targetID for _ in testLabels],
        "TestRun": [f"Run {nr + 1}" for _ in testLabels],
        "ProbSuccess": probSuccessSyn,
        "RecordPrivacyLossSyn": privLossSyn,
        "RecordPrivacyLossRaw": privLossRaw,
        "RecordPrivacyGain": privGain,
        "Accuracy": [get_attack_accuracy(tp, tn, len(rounded_prediction))],
        "Precision": [get_attack_precision(fp, tn)],
        "Recall": [get_attack_recall(tp, fn)],
        "AUC": [get_auc(testLabels, rounded_prediction)],
        "TestLabels": [testLabels],
        "RoundedPredictions": [rounded_prediction],
        "Predictions": [predictions],
    }

    return results


def worker_run_mia(params):
    (
        attacksList,
        target,
        targetID,
        rawWithoutTargets,
        attackDataIdx,
        testRawIndices,
        sizeRawTest,
        nShadows,
    ) = params

    # Generate shadow model data for training attacks on this target
    attackData = rawWithoutTargets.loc[attackDataIdx, :]

    synA, labelsSynA = generate_mia_shadow_data_shufflesplit(target, attackData, sizeRawTest, nShadows)

    for Attack in attacksList:
        Attack.train(synA, labelsSynA)

    # Clean up
    del synA, labelsSynA

    results = {
        AM.__name__: {
            "TargetID": [],
            "TestRun": [],
            "ProbSuccess": [],
            "RecordPrivacyLossSyn": [],
            "RecordPrivacyLossRaw": [],
            "RecordPrivacyGain": [],
            "Accuracy": [],
            "Precision": [],
            "Recall": [],
            "AUC": [],
            "TestLabels": [],
            "RoundedPredictions": [],
            "Predictions": [],
        }
        for AM in attacksList
    }

    for nr, rt in enumerate(testRawIndices):
        testRawOut = rawWithoutTargets.loc[rt, :]
        testDatawithoutTarget = [testRawOut]

        testRawIn = testRawOut.append(target)
        testDatawithTarget = [testRawIn]

        # Create balanced test dataset
        testData = testDatawithTarget + testDatawithoutTarget
        testLabels = [LABEL_IN for _ in range(len(testDatawithTarget))] + [
            LABEL_OUT for _ in range(len(testDatawithoutTarget))
        ]

        # Run attacks on synthetic datasets from target generative model
        for AM in attacksList:
            res = run_mia(AM, testData, testLabels, targetID, nr)
            for k, v in res.items():
                results[AM.__name__][k].extend(v)

        del testData, testLabels

    return results


def evaluate_mia(
    attacksList,
    rawWithoutTargets,
    targetRecords,
    targetIDs,
    attackDataIdx,
    testRawIndices,
    sizeRawTest,
    nShadows,
    nproc=multiprocessing.cpu_count(),
):
    # Train and test MIA per target
    print("Training attack classifiers...")
    t = time()
    with Pool(processes=nproc) as pool:
        tasks = [
            (
                attacksList,
                targetRecords.loc[[tid], :],
                tid,
                rawWithoutTargets,
                attackDataIdx,
                testRawIndices,
                sizeRawTest,
                nShadows,
            )
            for tid in targetIDs
        ]
        resultsList = []
        pbar = tqdm(total=len(tasks), smoothing=0)
        for res in pool.imap_unordered(worker_run_mia, tasks):
            resultsList.append(res)
            pbar.write("Accuracies: " + " ".join([f"{k}: {v['Accuracy']}" for k, v in res.items()]))
            pbar.update(1)
    print("Took:", time() - t)

    results = {
        AM.__name__: {
            "TargetID": [],
            "TestRun": [],
            "ProbSuccess": [],
            "RecordPrivacyLossSyn": [],
            "RecordPrivacyLossRaw": [],
            "RecordPrivacyGain": [],
            "Accuracy": [],
            "Precision": [],
            "Recall": [],
            "AUC": [],
            "TestLabels": [],
            "RoundedPredictions": [],
            "Predictions": [],
        }
        for AM in attacksList
    }

    for res in resultsList:
        for AM in attacksList:
            for k, v in res[AM.__name__].items():
                results[AM.__name__][k].extend(res[AM.__name__][k])

    for AM in attacksList:
        print()
        print(f"Attack {AM.__name__} with {len(targetRecords)} Targets")
        # print(f'Mean record privacy gain: {mean(results[AM.__name__]["RecordPrivacyGain"])}%')
        print(f'Mean accuracy : {mean(results[AM.__name__]["Accuracy"])}%')
        print(f'Mean precision : {mean(results[AM.__name__]["Precision"])}%')
        print(f'Mean recall : {mean(results[AM.__name__]["Recall"])}%')
        print(f'Mean AUC : {mean(results[AM.__name__]["AUC"])}%')
        print()

    return results


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--datapath", "-D", type=str, help="Path to local data file")
    argparser.add_argument(
        "--attack_model",
        "-AM",
        type=str,
        default="ANY",
        choices=["RandomForest", "LogReg", "LinearSVC", "SVC", "KNN", "ANY"],
    )
    argparser.add_argument("--outdir", "-O", default="output/mirage", type=str, help="Path for storing output files")
    args = argparser.parse_args()

    # Load runconfig
    runconfig = {
        "nIter": 25,
        "nTargets": 50,
        "sizeRawA": 500,
        "nShadows": 10,
        "sizeRawTest": 300,
        "sizeTargetCraft": 1,
        "prior": {"IN": 0.5, "OUT": 0.5},
    }
    print("Runconfig:")
    print(runconfig)

    # Load data
    RawDF, metadata = load_local_data_as_df(args.datapath)
    dname = args.datapath.split("/")[-1]
    RawDF["ID"] = [f"ID{i}" for i in arange(len(RawDF))]
    RawDF = RawDF.set_index("ID")

    print(f"Loaded data {dname}:")
    print(RawDF.info())

    # Randomly select nt target records T = (t_1, ..., t_(nt))
    targetIDs = choice(list(RawDF.index), size=runconfig["nTargets"], replace=False).tolist()
    Targets = RawDF.loc[targetIDs, :]

    # Drop targets from sample population
    RawDFdropT = RawDF.drop(targetIDs)

    # Add a crafted outlier target to the evaluation set
    targetCraft = craft_outlier(RawDF, runconfig["sizeTargetCraft"])
    targetIDs.extend(list(set(targetCraft.index)))
    Targets = Targets.append(targetCraft)

    # Sample adversary's background knowledge RawA
    attackDataIdx = choice(list(RawDFdropT.index), size=runconfig["sizeRawA"], replace=False).tolist()

    # Sample k independent target test sets
    testRawIndices = [
        choice(list(RawDFdropT.index), size=runconfig["sizeRawTest"], replace=False).tolist()
        for nr in range(runconfig["nIter"])
    ]

    FeatureList = [
        NaiveFeatureSet(DataFrame),
        HistogramFeatureSet(DataFrame, metadata),
        CorrelationsFeatureSet(DataFrame, metadata),
        EnsembleFeatureSet(DataFrame, metadata),
    ]

    prior = {LABEL_IN: runconfig["prior"]["IN"], LABEL_OUT: runconfig["prior"]["OUT"]}
    if args.attack_model == "RandomForest":
        AttacksList = [MIAttackClassifierRandomForest(metadata, prior, F) for F in FeatureList]
    elif args.attack_model == "LogReg":
        AttacksList = [MIAttackClassifierLogReg(metadata, prior, F) for F in FeatureList]
    elif args.attack_model == "LinearSVC":
        AttacksList = [MIAttackClassifierLinearSVC(metadata, prior, F) for F in FeatureList]
    elif args.attack_model == "SVC":
        AttacksList = [MIAttackClassifierSVC(metadata, prior, F) for F in FeatureList]
    elif args.attack_model == "KNN":
        AttacksList = [MIAttackClassifierKNN(metadata, prior, F) for F in FeatureList]
    elif args.attack_model == "ANY":
        AttacksList = []
        for F in FeatureList:
            AttacksList.extend(
                [
                    MIAttackClassifierRandomForest(metadata, prior, F),
                    MIAttackClassifierLogReg(metadata, prior, F),
                    MIAttackClassifierKNN(metadata, prior, F),
                ]
            )
    else:
        raise ValueError(f"Unknown AM {args.attack_model}")

    # Run privacy evaluation under MIA adversary
    results = evaluate_mia(
        AttacksList,
        RawDFdropT,
        Targets,
        targetIDs,
        attackDataIdx,
        testRawIndices,
        runconfig["sizeRawTest"],
        runconfig["nShadows"],
    )

    outfile = f"{dname}-MIA"
    with open(path.join(f"{args.outdir}", f"{outfile}.json"), "w") as f:
        json.dump(results, f)

    if not os.path.exists(path.join(f"{args.outdir}/{dname}")):
        os.mkdir(path.join(f"{args.outdir}/{dname}"))

    for key in list(results.keys()):
        plot_roc(results[key]["TestLabels"], results[key]["Predictions"], path.join(f"{args.outdir}/{dname}", f"{key}"))
        plot_hist(
            results[key]["TestLabels"], results[key]["Predictions"], path.join(f"{args.outdir}/{dname}", f"{key}")
        )
