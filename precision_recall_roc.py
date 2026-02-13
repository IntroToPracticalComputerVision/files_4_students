import pickle

import numpy as np
import pandas as pd

# from matplotlib import pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, average_precision_score




def generate_precision_recall_auc_graphs(y_true, y_predict, version, show_threshold=True):
    #---------------------------------------------------------lab3---------------------------------------------------
    # get the precision, recall and thresholds using precision_recall_curve function[1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0]
    recall, precision, thresholds_pr  = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] # fix this line!!! use drop_intermediate=False
    #---------------------------------------------------------lab3---------------------------------------------------


    # get the average precision  (AP) using average_precision_score
    AP = 0
    # plt.figure('precision recall')
    # plt.title('precision recall')
    # plt.plot(recall1, precision1, label=version + f' AP ={np.round(AP, decimals=2)}')
    #
    # plt.xlabel('Recall')
    #
    # plt.ylabel('Precision')

    labels = [str(float(int(i * 100)) / 100) for i in thresholds_pr]

    labels += ['inf']

    # Annotate each point with its corresponding labels
    points_to_show = 5
    jump_every = int(len(precision) / points_to_show)
    if show_threshold:
        for i, (xi, yi) in enumerate(zip(recall, precision)):
            # print(i)
            if i % jump_every==0:
                label = ''.join(labels[i])  # Convert the list of strings to a single string separated by newlines

                # plt.annotate(label, (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center')
    # plt.legend()

    #---------------------------------------------------------lab3---------------------------------------------------
    # get the fpr, tpr and thresholds2 using roc_curve function
    fpr, tpr, thresholds_roc = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] # fix this line!!! use drop_intermediate=False
    #---------------------------------------------------------lab3---------------------------------------------------

    # get the auc using roc_auc_score
    auc = 0

    # plt.figure('ROC curve')
    # plt.title('ROC curve')
    #
    # plt.plot(fpr, tpr, label=version + f' Auc ={np.round(auc, decimals=2)}')
    #
    # plt.xlabel('fpr')
    #
    # plt.ylabel('tpr')

    labels = [str(float(int(i * 100)) / 100) for i in thresholds_roc[1:]]

    labels = ['inf'] + labels
    jump_every = int(len(fpr) / points_to_show)

    # Annotate each point with its corresponding labels
    if show_threshold:
        for i, (xi, yi) in enumerate(zip(fpr, tpr)):
            # print(i)
            if i % jump_every==0:
                label = ''.join(labels[i])  # Convert the list of strings to a single string separated by newlines

                # plt.annotate(label, (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center')

    return recall, precision, thresholds_pr, fpr, tpr, thresholds_roc

if __name__ == "__main__":
    y_true = [1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0]

    y_predict = ((100 * np.arange(1, -1 / (len(y_true) - 1), -1 / (len(y_true) - 1))).astype(np.int32)).astype(
        np.float32) / 100
    generate_precision_recall_auc_graphs(y_true, y_predict, version='v1', show_threshold=False)
    y_true = [1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    generate_precision_recall_auc_graphs(y_true, y_predict, version='v2', show_threshold=False)

    plt.legend()
    # plt.pause(0.1)
    # plt.show()
    plt.figure('precision recall')
    plt.savefig('precision_recall_example.png')  # Change the extension to save in different formats, e.g., .pdf, .svg
    plt.figure('ROC curve')
    plt.savefig('roc_curve_example.png')  # Change the extension to save in different formats, e.g., .pdf, .svg

    data_base = pd.read_csv('gt_tracked_objects.csv')

    y_true = data_base.tag.values
    y_predict = data_base.score.values.tolist()
    y_predict = (y_predict/np.max(y_predict)).tolist()
    generate_precision_recall_auc_graphs(y_true, y_predict, version='v1', show_threshold=True)

    plt.legend()
    plt.pause(0.1)
    plt.show()
    plt.figure('precision recall')
    plt.savefig('gt_precision_recall.png')  # Change the extension to save in different formats, e.g., .pdf, .svg
    plt.figure('ROC curve')
    plt.savefig('gt_roc_curve.png')  # Change the extension to save in different formats, e.g., .pdf, .svg

