import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve, classification_report
import json
import statistics


def plot_hist(target, prob_succes, name):
    font = {"weight": "bold", "size": 10}
    matplotlib.rc("font", **font)

    mpreds = []
    nmpreds = []

    for i in range(0, len(target)):
        for j in range(0, len(target[i])):
            if target[i][j] == 1:
                mpreds.append(prob_succes[i][j][1])
            else:
                nmpreds.append(prob_succes[i][j][1])

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
    plt.savefig(f"results/endterm/{name}_privacy_risk.png")
    plt.close()


def plot_roc(target, probs, name):
    new_target = []
    new_probs = []
    for i in range(0, len(target)):
        for j in range(0, len(target[i])):
            new_target.append(target[i][j])
            new_probs.append(probs[i][j][1])
    font = {"weight": "bold", "size": 10}
    matplotlib.rc("font", **font)
    fpr, tpr, _ = roc_curve(new_target, new_probs)
    roc_auc = auc(fpr, tpr)
    plt.title("ROC of Membership Inference Attack")
    plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.savefig(f"results/endterm/{name}_roc.png")
    plt.close()


def plot_nasr(target, probs, name):
    font = {"weight": "bold", "size": 10}
    mpreds = probs[target == 1]
    nmpreds = probs[target == 0]

    matplotlib.rc("font", **font)

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
    plt.savefig(f"results/endterm/{name}_privacy_risk.png")
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
    plt.savefig(f"results/endterm/{name}_roc.png")
    plt.close()

def plot_nasr_multiple_roc(labels, target, probs, roc_info, name):
    print(labels)
    print(int(roc_info[0][0]))
    probs_min = probs[labels == int(roc_info[0][0])]
    target_min = target[labels == int(roc_info[0][0])]
    probs_median = probs[labels == int(roc_info[1][0])]
    target_median = target[labels == int(roc_info[1][0])]
    probs_max = probs[labels == int(roc_info[2][0])]
    target_max = target[labels == int(roc_info[2][0])]
    print("CHECK1:")
    print(len(probs_min))
    print(len(target_median))
    print(len(probs_max))

    # Creates ROC curve for membership inference attack
    fpr_min, tpr_min, _ = roc_curve(target_min, probs_min)
    roc_auc_min = auc(fpr_min, tpr_min)
    fpr_median, tpr_median, _ = roc_curve(target_median, probs_median)
    roc_auc_median = auc(fpr_median, tpr_median)
    fpr_max, tpr_max, _ = roc_curve(target_max, probs_max)
    roc_auc_max = auc(fpr_max, tpr_max)
    plt.title("ROC of Membership Inference Attack")
    plt.plot(fpr_min, tpr_min, "b", label="AUC = %0.2f" % roc_auc_min)
    plt.plot(fpr_median, tpr_median, "b", label="AUC = %0.2f" % roc_auc_median)
    plt.plot(fpr_max, tpr_max, "b", label="AUC = %0.2f" % roc_auc_max)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.savefig(f"results/endterm/{name}_multiple_roc.png")
    plt.close()

def save_classification_report(y_true, y_pred, name):
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(f"results/endterm/{name}_classification_report.json", 'w') as outfile:
        json.dump(report, outfile)

    fscores = []
    support = []
    keys = list(report.keys())
    print("CHECK keys:")
    print(keys)
    for key in keys:
        if key.isdigit() and report[key]["support"] > 0:
            fscores.append(report[key]["f1-score"])
            support.append(report[key]["support"])

    maximum = max(fscores)
    minimum = min(fscores)
    median = statistics.median(fscores)
    max_indexes = []
    min_indexes = []
    median_indexes = []
    for i in range(len(fscores)):
        i = int(i)
        if fscores[i] == maximum:
            max_indexes.append(i)
        if fscores[i] == minimum:
            min_indexes.append(i)
        if fscores[i] == median:
            median_indexes.append(i)

    max_index = max_indexes[0]
    min_index = min_indexes[0]
    median_index = min_indexes[0]

    support_count = 0
    for i in max_indexes:
        if support[i] > support_count:
            support_count = support[i]
            max_index = i
    support_count = 0
    for i in min_indexes:
        if support[i] > support_count:
            support_count = support[i]
            min_index = i
    support_count = 0
    for i in median_indexes:
        if support[i] > support_count:
            support_count = support[i]
            median_index = i

    return [[keys[min_index], report[keys[min_index]]["f1-score"]], [keys[median_index], report[keys[median_index]]["f1-score"]], [keys[max_index], report[keys[max_index]]["f1-score"]]]



if __name__ == "__main__":

    file = "./output/mirage/texas-MIA.json"
    f = open(file, "r")
    results = json.load(f)
    for key in list(results.keys()):
        plot_roc(results[key]["TestLabels"], results[key]["Predictions"], key)
        plot_hist(results[key]["TestLabels"], results[key]["Predictions"], key)
