import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


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


if __name__ == "__main__":
    import json

    file = "./output/mirage/texas-MIA.json"
    f = open(file, "r")
    results = json.load(f)
    for key in list(results.keys()):
        plot_roc(results[key]["TestLabels"], results[key]["Predictions"], key)
        plot_hist(results[key]["TestLabels"], results[key]["Predictions"], key)
