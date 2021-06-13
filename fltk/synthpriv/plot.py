import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve, classification_report
import json


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
    probs_min = [probs[i] for i, x in enumerate(labels) if x == int(roc_info[0][0])]
    target_min = [target[i] for i, x in enumerate(labels) if x == int(roc_info[0][0])]
    probs_median = [probs[i] for i, x in enumerate(labels) if x == int(roc_info[1][0])]
    target_median = [target[i] for i, x in enumerate(labels) if x == int(roc_info[1][0])]
    probs_max = [probs[i] for i, x in enumerate(labels) if x == int(roc_info[2][0])]
    target_max = [target[i] for i, x in enumerate(labels) if x == int(roc_info[2][0])]
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
    plt.plot(fpr_min, tpr_min, label="Min = %f, AUC = %0.2f" % (roc_info[0][1], roc_auc_min))
    plt.plot(fpr_median, tpr_median, label="Median= %f, AUC = %0.2f" % (roc_info[1][1], roc_auc_median))
    plt.plot(fpr_max, tpr_max, label="Max = %f, AUC = %0.2f" % (roc_info[2][1], roc_auc_max))
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.savefig(f"results/endterm/{name}_multiple_roc.png")
    plt.close()


def get_median(lst):
    sortedLst = sorted(lst)
    lstLen = len(lst)
    index = (lstLen - 1) // 2

    return sortedLst[index]


def save_classification_report(y_true, y_pred, name):
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(f"results/endterm/{name}_classification_report.json", 'w') as outfile:
        json.dump(report, outfile, indent=2)

    fscores = {}
    support = {}
    keys = list(report.keys())
    for key in keys:
        if key.isdigit() and report[key]["support"] > 0:
            fscores[key] = report[key]["f1-score"]
            support[key] = report[key]["support"]

    fscores_list = list(fscores.values())
    maximum = max(fscores_list)
    minimum = min(fscores_list)
    median = get_median(fscores_list)
    max_keys = []
    min_keys = []
    median_keys = []
    for i in list(fscores.keys()):
        if fscores[i] == maximum:
            max_keys.append(i)
        if fscores[i] == minimum:
            min_keys.append(i)
        if fscores[i] == median:
            median_keys.append(i)
    max_key = max_keys[0]
    min_key = min_keys[0]
    median_key = median_keys[0]

    support_count = 0
    for i in max_keys:
        if support[i] > support_count:
            support_count = support[i]
            max_key = i
    support_count = 0
    for i in min_keys:
        if support[i] > support_count:
            support_count = support[i]
            min_key = i
    support_count = 0
    for i in median_keys:
        if support[i] > support_count:
            support_count = support[i]
            median_key = i

    return [[min_key, report[min_key]["f1-score"]], [median_key, report[median_key]["f1-score"]], [max_key, report[max_key]["f1-score"]]]



if __name__ == "__main__":
    file = "./output/mirage/texas-MIA.json"
    f = open(file, "r")
    results = json.load(f)
    for key in list(results.keys()):
        plot_roc(results[key]["TestLabels"], results[key]["Predictions"], key)
        plot_hist(results[key]["TestLabels"], results[key]["Predictions"], key)
