# import os
#
# import pandas as pd
# import sklearn.neighbors as skn
# import torch
#
# from src.data.loss_dataset import loss_dataset
#
#
# def assign_metadata(
#     dataset: str, input_filepath: str, output_filepath: str, classifier
# ) -> None:
#     """
#     Assigns metadata to the dataset.
#
#     Args:
#         dataset (str): Name of the dataset.
#         output_filepath (str): Path to the output directory.
#         clf: Classifier to use for assigning metadata.
#     """
#     losses = loss_dataset(dataset)
#     probes = pd.read_csv(os.path.join(input_filepath, dataset, "probes.csv"))
#     clf = classifier
#
#
# assign_metadata("cifar10", "ost", skn.KNeighborsClassifier())
