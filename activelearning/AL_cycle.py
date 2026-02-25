import logging
from collections.abc import Callable
from typing import Any, Literal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from skorch import NeuralNetClassifier
from torch import nn

from activelearning.queries.informative.entropy_query_stream import stream_query_entropy
from activelearning.queries.informative.margin_query_stream import stream_query_margin
from activelearning.queries.informative.uncertainty_query_stream import stream_query_uncertainty
from activelearning.queries.representative.coreset_query import query_coreset
from activelearning.queries.representative.coreset_query_stream import stream_query_coreset
from activelearning.queries.representative.diversity_query_stream import stream_query_diversity
from activelearning.queries.representative.probcover_query_stream import stream_query_probcover
from activelearning.queries.representative.random_query import query_random
from activelearning.queries.representative.random_query_stream import stream_query_random
from activelearning.utils.AL_utils import (
    ValidClassifiers,
    score_accuracy,
    set_committee,
    set_learner,
    set_query_args,
    set_query_names,
)
from activelearning.utils.skorch_nnet import (
    reshapedVGG,
)

import copy
from sklearn.metrics import f1_score
from skorch.dataset import CVSplit
import os
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D


from src.utils import get_embeddings


class SkorchActiveLearner(ActiveLearner):
    def teach(self, X, y, only_new=False, checkpoint_name=None, **fit_kwargs):
        if self.X_training is None or only_new:
            self.X_training = X
            self.y_training = y
        else:
            try:
                self.X_training = np.concatenate((self.X_training, X), axis=0)
                self.y_training = np.concatenate((self.y_training, y))
            except ValueError:
                raise ValueError('the dimensions of the new training data and label must'
                                 'agree with the training data and labels provided so far')

        # self.estimator.train_split = CVSplit(cv=0.15 / 0.85, stratified=True, random_state=42)
        
        self.estimator.fit(self.X_training, self.y_training, **fit_kwargs)
        # self.estimator.load_params(f_params='../examples/best_weights.pt')
        self.estimator.load_params(f_params=checkpoint_name)
        

def plot_tsne_selections(classifier, X_pool, y_pool, query_idx, strategy_name, round_num, output_dir="Ablation_study"):
    """
    Reduces the embeddings of the current pool using t-SNE and plots them.
    Includes dual legends: one for Sample Type and one for Classes.
    """
    print(f"[INFO] Generating t-SNE plot for {strategy_name} - Round {round_num}...")
    
    # 1. Get embeddings
    pool_embeddings_tensor, _ = get_embeddings(classifier, X_pool)
    pool_embeddings = pool_embeddings_tensor.numpy()
    
    # 2. Fit t-SNE (reduces 64D to 2D)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(pool_embeddings)
    
    # 3. Create the plot (made slightly wider to fit the outside legend)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot unlabeled pool
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=y_pool, cmap='tab10', alpha=0.3, s=20)
    
    # Plot queried samples
    ax.scatter(embeddings_2d[query_idx, 0], embeddings_2d[query_idx, 1], 
               c='red', marker='*', s=150, edgecolor='black')
    
    # --- LEGEND 1: Sample Type (Inside the plot) ---
    marker_unlabeled = Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, alpha=0.5, label='Unlabeled Pool')
    marker_queried = Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markeredgecolor='black', markersize=15, label='Queried Samples')
    
    # legend1 = ax.legend(handles=[marker_unlabeled, marker_queried], loc='upper right', title="Sample Type", title_fontproperties={'weight':'bold'})
    # ax.add_artist(legend1) # This line ensures Legend 1 isn't erased when we add Legend 2
    
    # --- LEGEND 2: Classes (Outside the plot) ---
    unique_classes = np.unique(y_pool)
    class_handles = []
    
    # You can map these to actual string names later if you want 
    # (e.g., 0: 'tightening', 1: 'untightening', etc.) based on your label_map
    for cls in unique_classes:
        # Extract the exact color used for this class from the scatter plot
        color = scatter.cmap(scatter.norm(cls))
        class_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=f'Class {int(cls)}'))
        
    # Place this legend outside the plot box so it doesn't cover points
    ax.legend(handles=class_handles, loc='center left', bbox_to_anchor=(1.02, 0.5), title="Action Classes", title_fontproperties={'weight':'bold'})
    
    # 4. Formatting and Saving
    clean_name = strategy_name.__name__ if hasattr(strategy_name, '__name__') else str(strategy_name)
    plt.title(f"t-SNE Embeddings: {clean_name} (Round {round_num})", fontsize=16, pad=15)
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"tsne_{clean_name}_round_{round_num}.png")
    
    # bbox_inches='tight' ensures the outside legend isn't cut off when saving
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved t-SNE plot to {save_path}")

# def cycle_AL(
#     X_train: np.ndarray | None,
#     y_train: np.ndarray | None,
#     X_pool: np.ndarray,
#     y_pool: np.ndarray,
#     X_test: np.ndarray,
#     y_test: np.ndarray,
#     X_pool_org: np.ndarray | None,
#     X_train_org: np.ndarray | None,
#     classifier: Any,
#     query_strategy: Callable,
#     committee_classifiers: list | None = None,
#     n_instances: int = 1,
#     batch_size: int = 0,
#     start_len: int = 50,
#     K: int = 3,
#     metric: str = "euclidean",
#     input_dim: int = 512,
#     output_dim: int = 10,
#     max_epochs: int = 50,
#     quantile: float = 0.5,
#     goal_metric: str = "f1",
#     goal_metric_val: float = 0.75,
#     acc: str = "test",
# ) -> tuple[list, list]:
#     """Executes Active Learning on provided data with the specified strategy and parameters.

#     Note:
#         If batch_size is specified, then it become stream based batch sampling.
#         At every iteration, saves a batch of points, performs the query strategy on this batch, then discards the
#         points that are not chosen.

#     Inputs:
#         X_train: initial labeled data
#         y_train: labels for X_train
#             both can be "None" if starting without labeled data
#         X_pool: initial unlabeled data
#         y_pool: labels for X_pool
#         X_test: test data
#         y_test_ lebels for test data
#         classifier: model type used for classification
#             supports "randomforest", "nnet" and the other values in ValidClassifiers, or an already instantiated model
#         committee_classifiers: list of classifiers to be used for the committee. If None, no committee is used
#         query_strategy: one the implemented query strategies imported from the queries folder.
#         n_instances: number of instances to be queried at each iteration
#         batch_size: if left at 0 is ignored, otherwise the size of the batch for stream based batch sampling
#         start_len: for stream based sampling, number of points to be kept as initial set

#         K: number of clusters, only if query strategy is k-means
#         metric: density metric, only if query strategy is density
#         quantile: quantile of distances for stream based diversity query threshold

#         input_dim: size of inputs, in case a nnet classifier is passed
#         output_dim: number of classes, in case a nnet classifier is passed
#         max_epochs: max number of epochs for the training of a nnet at every iteration
#         goal_acc: querying continues until this accuracy is reached, or all the pool has been sampled
#         acc: whether to asses the classification accuracy on test set, remaining pool, or joint.

#     Outputs:
#         acc_list: list of accuracy scores, one for each iteration
#         inst_list: keeps track of how many instances have been sampled after each iteration
#     """

#     if X_pool_org is None:
#         X_pool_org = X_pool

#     if X_train_org is None:
#         X_train_org = X_train


#     if committee_classifiers is None:
#         committee_classifiers = []
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # check if is stream based batch sampling or not
#     if batch_size == 0:
#         is_stream = False
#     else:
#         is_stream = True

#     # declare learner as ActiveLearner class
#     valid_classifiers = [c.value for c in ValidClassifiers]
#     if classifier in valid_classifiers:
#         learner = set_learner(classifier, query_strategy, X_train_org, y_train, input_dim, output_dim, max_epochs, device)
#     else:
#         # in this case the classifier is assumed to be a non-default sklearn classifier
#         # if the model is wrongly specified, activelearner will give an error
#         # learner = ActiveLearner(
#         #     estimator=classifier, query_strategy=query_strategy, X_training=X_train_org, y_training=y_train
#         # )
#         learner = SkorchActiveLearner(
#             estimator=classifier, query_strategy=query_strategy, X_training=X_train_org, y_training=y_train
#         )

#         classifier_copy = copy.deepcopy(classifier)

#     if committee_classifiers != []:
#         committee = set_committee(
#             committee_classifiers, query_strategy, X_train, y_train, input_dim, output_dim, max_epochs, device
#         )

#     if is_stream is True:
#         # initialize first batch
#         X_pool_full, y_pool_full = X_pool, y_pool
#         if X_pool_full.shape[0] < batch_size:
#             batch_size = X_pool_full.shape[0]
#         batch_idxs = np.random.choice(range(X_pool_full.shape[0]), size=batch_size, replace=False)
#         X_pool, y_pool = X_pool_full[batch_idxs], y_pool[batch_idxs]
#         X_pool_full = np.delete(X_pool_full, batch_idxs, axis=0)
#         y_pool_full = np.delete(y_pool_full, batch_idxs, axis=0)

#     # initialize query args dict if empty
#     query_args = set_query_args(
#         learner.estimator,
#         query_strategy,
#         X_pool,
#         y_pool,
#         # learner.X_training,
#         X_train,
#         n_instances,
#         K,
#         metric,
#         quantile,
#         start_len,
#     )

#     n_instances = query_args["n_instances"]
#     if X_train is None:
#         instances_sampled = 0
#     else:
#         instances_sampled = X_train.shape[0]

#     # init acc_list and inst_list, and calculate starting values if there's a starting set
#     accuracy = 0
#     f1_scr = 0
#     inst_list = []
#     acc_list = []
#     f1_list = []
#     if X_train is not None:
#         accuracy = score_accuracy(learner, acc, X_pool_org, y_pool, X_test, y_test)
#         acc_list.append(accuracy)

#         y_pred = learner.predict(X_test)
#         f1_scr = f1_score(y_test, y_pred, average="macro")
#         f1_list.append(f1_scr)

#         inst_list.append(instances_sampled)

#     metric_val = 0.0
#     if goal_metric == "f1":
#         metric_val = f1_scr
#     elif goal_metric == "acc":
#         metric_val = accuracy
        
#     # iterates until goal accuracy is reached or all pool is sampled
#     # while accuracy < goal_acc and X_pool.shape[0] > 0:
#     while metric_val < goal_metric_val and X_pool.shape[0] > 0:

#         if X_pool.shape[0] - n_instances < 0:
#             # if there's less instances left, sample all of them
#             n_instances = X_pool.shape[0]
#             query_args["n_instances"] = n_instances

#         stream_queries = [
#             stream_query_diversity,
#             stream_query_coreset,
#             stream_query_probcover,
#             stream_query_random,
#             stream_query_uncertainty,
#             stream_query_margin,
#             stream_query_entropy,
#         ]
#         if query_strategy in stream_queries:  # stream based query
#             query_idx, _, kept = query_strategy(**query_args)
#         elif committee_classifiers != []:  # committee based query
#             query_idx, _ = committee.query(**query_args)
#             kept = True
#         else:  # pool based query
#             query_idx, _ = learner.query(**query_args)
#             kept = True

#         if kept is True:
#             if n_instances == 1:
#                 shape = n_instances
#             else:
#                 shape = len(query_idx)
#             if classifier == "nnet_bo":  # re-instantiate the classifier
#                 # just initializing doesn't work
#                 learner.estimator = NeuralNetClassifier(
#                     reshapedVGG(num_classes=output_dim),
#                     max_epochs=max_epochs,
#                     criterion=nn.CrossEntropyLoss,
#                     optimizer=torch.optim.Adam,
#                     train_split=None,
#                     verbose=0,
#                     device=device,
#                 )
            
#             learner.estimator = classifier_copy
#             classifier_copy = copy.deepcopy(classifier_copy)

#             learner.teach(
#                 # X_pool[query_idx].reshape(shape, -1),
#                 X_pool_org[query_idx],
#                 y_pool[query_idx],
#                 only_new=False,
#             )  # appends instances to labeled set
#             if committee_classifiers != []:
#                 committee.teach(
#                     X_pool[query_idx].reshape(shape, -1),
#                     y_pool[query_idx].reshape(
#                         shape,
#                     ),
#                 )

#         if "X_start" in query_args:
#             # query_args["X_start"] = np.concat((query_args["X_start"], X_pool[query_idx]), axis=0) if "X_pool_emb" in query_args else learner.X_training
#             query_args["X_start"] = np.concat((query_args["X_start"], X_pool[query_idx]), axis=0)

#         if is_stream is False:
#             # remove queried instance from the pool
#             X_pool = np.delete(X_pool, query_idx, axis=0)
#             X_pool_org = np.delete(X_pool_org, query_idx, axis=0)
#             y_pool = np.delete(y_pool, query_idx)

#         else:  # stream based, prepare next batch
#             if X_pool_full.shape[0] < batch_size:
#                 batch_size = X_pool_full.shape[0]
#             batch_idxs = np.random.choice(X_pool_full.shape[0], size=batch_size, replace=False)

#             X_pool, y_pool = X_pool_full[batch_idxs], y_pool_full[batch_idxs]
#             X_pool_full = np.delete(X_pool_full, batch_idxs, axis=0)
#             y_pool_full = np.delete(y_pool_full, batch_idxs, axis=0)

#         # update input dictionary for the query function
#         query_args["X_pool"] = X_pool

#         # update other parameters of query strategy
#         # if "X_start" in query_args:
#         #     # query_args["X_start"] = np.concat((query_args["X_start"], X_pool[query_idx]), axis=0) if "X_pool_emb" in query_args else learner.X_training
#         #     query_args["X_start"] = np.concat((query_args["X_start"], X_pool[query_idx]), axis=0)
#         if "classifier" in query_args:
#             query_args["classifier"] = learner.estimator

#         if kept is True:
#             accuracy = score_accuracy(learner, acc, X_pool_org, y_pool, X_test, y_test)
#             acc_list.append(accuracy)
            
#             y_pred = learner.predict(X_test)
#             f1_scr = f1_score(y_test, y_pred, average="macro")
#             f1_list.append(f1_scr)

#             if goal_metric == "f1":
#                 metric_val = f1_scr
#             elif goal_metric == "acc":
#                 metric_val = accuracy

#             instances_sampled += n_instances
#             inst_list.append(instances_sampled)

#     return acc_list, f1_list, inst_list



# def get_embeddings(classifier: Any, X: np.ndarray):
#     net = classifier.module_

#     # Save state
#     prev_mode = net.output_mode
#     was_training = net.training

#     net.output_mode = "embedding"
#     net.eval()

#     embeddings = []
#     probs = []

#     with torch.no_grad():
#         for emb, logit in classifier.forward_iter(X, training=False):
#             embeddings.append(emb.detach().cpu())
#             probs.append(F.softmax(logit, dim=-1).detach().cpu())

#     # Restore state
#     net.output_mode = prev_mode
#     net.train(was_training)

#     return torch.cat(embeddings, dim=0), torch.cat(probs, dim=0)


# def cycle_AL(
#     X_train: np.ndarray | None,
#     y_train: np.ndarray | None,
#     # X_pool: np.ndarray,
#     y_pool: np.ndarray,
#     X_pool_org: np.ndarray | None,
#     X_test: np.ndarray,
#     y_test: np.ndarray,
#     # X_train_org: np.ndarray | None,
#     classifier: Any,
#     query_type: Literal["uncertainty", "diversity"],
#     query_strategy: Callable,
#     committee_classifiers: list | None = None,
#     n_instances: int = 1,
#     batch_size: int = 0,
#     start_len: int = 50,
#     K: int = 3,
#     metric: str = "euclidean",
#     input_dim: int = 512,
#     output_dim: int = 10,
#     max_epochs: int = 50,
#     quantile: float = 0.5,
#     goal_metric: str = "f1",
#     goal_metric_val: float = 0.75,
#     acc: str = "test",
# ) -> tuple[list, list]:
#     """Executes Active Learning on provided data with the specified strategy and parameters.

#     Note:
#         If batch_size is specified, then it become stream based batch sampling.
#         At every iteration, saves a batch of points, performs the query strategy on this batch, then discards the
#         points that are not chosen.

#     Inputs:
#         X_train: initial labeled data
#         y_train: labels for X_train
#             both can be "None" if starting without labeled data
#         X_pool: initial unlabeled data
#         y_pool: labels for X_pool
#         X_test: test data
#         y_test_ lebels for test data
#         classifier: model type used for classification
#             supports "randomforest", "nnet" and the other values in ValidClassifiers, or an already instantiated model
#         committee_classifiers: list of classifiers to be used for the committee. If None, no committee is used
#         query_strategy: one the implemented query strategies imported from the queries folder.
#         n_instances: number of instances to be queried at each iteration
#         batch_size: if left at 0 is ignored, otherwise the size of the batch for stream based batch sampling
#         start_len: for stream based sampling, number of points to be kept as initial set

#         K: number of clusters, only if query strategy is k-means
#         metric: density metric, only if query strategy is density
#         quantile: quantile of distances for stream based diversity query threshold

#         input_dim: size of inputs, in case a nnet classifier is passed
#         output_dim: number of classes, in case a nnet classifier is passed
#         max_epochs: max number of epochs for the training of a nnet at every iteration
#         goal_acc: querying continues until this accuracy is reached, or all the pool has been sampled
#         acc: whether to asses the classification accuracy on test set, remaining pool, or joint.

#     Outputs:
#         acc_list: list of accuracy scores, one for each iteration
#         inst_list: keeps track of how many instances have been sampled after each iteration
#     """

#     # if X_pool_org is None:
#     #     X_pool_org = X_pool

#     # if X_train_org is None:
#     #     X_train_org = X_train


#     if committee_classifiers is None:
#         committee_classifiers = []
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # check if is stream based batch sampling or not
#     if batch_size == 0:
#         is_stream = False
#     else:
#         is_stream = True

#     # declare learner as ActiveLearner class
#     valid_classifiers = [c.value for c in ValidClassifiers]
#     if classifier in valid_classifiers:
#         learner = set_learner(classifier, query_strategy, X_train, y_train, input_dim, output_dim, max_epochs, device)
#     else:
#         # in this case the classifier is assumed to be a non-default sklearn classifier
#         # if the model is wrongly specified, activelearner will give an error
#         # learner = ActiveLearner(
#         #     estimator=classifier, query_strategy=query_strategy, X_training=X_train_org, y_training=y_train
#         # )
#         learner = SkorchActiveLearner(
#             estimator=classifier, query_strategy=query_strategy, X_training=X_train, y_training=y_train
#         )

#         classifier_copy = copy.deepcopy(classifier)

#     if query_type == "diversity":
#         X_start, _ = get_embeddings(learner.estimator, learner.X_training)
#         X_pool, _ = get_embeddings(learner.estimator, X_pool_org)

#     elif query_type == "uncertainty":
#         X_start = learner.X_training
#         X_pool = X_pool_org


#     if committee_classifiers != []:
#         committee = set_committee(
#             committee_classifiers, query_strategy, X_train, y_train, input_dim, output_dim, max_epochs, device
#         )

#     if is_stream is True:
#         # initialize first batch
#         X_pool_full, y_pool_full = X_pool, y_pool
#         if X_pool_full.shape[0] < batch_size:
#             batch_size = X_pool_full.shape[0]
#         batch_idxs = np.random.choice(range(X_pool_full.shape[0]), size=batch_size, replace=False)
#         X_pool, y_pool = X_pool_full[batch_idxs], y_pool[batch_idxs]
#         X_pool_full = np.delete(X_pool_full, batch_idxs, axis=0)
#         y_pool_full = np.delete(y_pool_full, batch_idxs, axis=0)

#     # initialize query args dict if empty
#     query_args = set_query_args(
#         learner.estimator,
#         query_strategy,
#         X_pool,
#         y_pool,
#         # learner.X_training,
#         X_start,
#         n_instances,
#         K,
#         metric,
#         quantile,
#         start_len,
#     )

#     n_instances = query_args["n_instances"]
#     if X_train is None:
#         instances_sampled = 0
#     else:
#         instances_sampled = X_train.shape[0]

#     # init acc_list and inst_list, and calculate starting values if there's a starting set
#     accuracy = 0
#     f1_scr = 0
#     inst_list = []
#     acc_list = []
#     f1_list = []
#     if X_train is not None:
#         accuracy = score_accuracy(learner, acc, X_pool_org, y_pool, X_test, y_test)
#         acc_list.append(accuracy)

#         y_pred = learner.predict(X_test)
#         f1_scr = f1_score(y_test, y_pred, average="macro")
#         f1_list.append(f1_scr)

#         inst_list.append(instances_sampled)

#     metric_val = 0.0
#     if goal_metric == "f1":
#         metric_val = f1_scr
#     elif goal_metric == "acc":
#         metric_val = accuracy
        
#     # iterates until goal accuracy is reached or all pool is sampled
#     # while accuracy < goal_acc and X_pool.shape[0] > 0:
#     while metric_val < goal_metric_val and X_pool.shape[0] > 0:

#         if X_pool.shape[0] - n_instances < 0:
#             # if there's less instances left, sample all of them
#             n_instances = X_pool.shape[0]
#             query_args["n_instances"] = n_instances

#         stream_queries = [
#             stream_query_diversity,
#             stream_query_coreset,
#             stream_query_probcover,
#             stream_query_random,
#             stream_query_uncertainty,
#             stream_query_margin,
#             stream_query_entropy,
#         ]
#         if query_strategy in stream_queries:  # stream based query
#             query_idx, _, kept = query_strategy(**query_args)
#         elif committee_classifiers != []:  # committee based query
#             query_idx, _ = committee.query(**query_args)
#             kept = True
#         else:  # pool based query
#             query_idx, _ = learner.query(**query_args)
#             kept = True

#         if kept is True:
#             if n_instances == 1:
#                 shape = n_instances
#             else:
#                 shape = len(query_idx)
#             if classifier == "nnet_bo":  # re-instantiate the classifier
#                 # just initializing doesn't work
#                 learner.estimator = NeuralNetClassifier(
#                     reshapedVGG(num_classes=output_dim),
#                     max_epochs=max_epochs,
#                     criterion=nn.CrossEntropyLoss,
#                     optimizer=torch.optim.Adam,
#                     train_split=None,
#                     verbose=0,
#                     device=device,
#                 )
            
#             learner.estimator = classifier_copy
#             classifier_copy = copy.deepcopy(classifier_copy)

#             learner.teach(
#                 # X_pool[query_idx].reshape(shape, -1),
#                 X_pool_org[query_idx],
#                 y_pool[query_idx],
#                 only_new=False,
#             )  # appends instances to labeled set
#             if committee_classifiers != []:
#                 committee.teach(
#                     X_pool[query_idx].reshape(shape, -1),
#                     y_pool[query_idx].reshape(
#                         shape,
#                     ),
#                 )

#         if "X_start" in query_args:
#             # query_args["X_start"] = np.concat((query_args["X_start"], X_pool[query_idx]), axis=0) if "X_pool_emb" in query_args else learner.X_training
#             # query_args["X_start"] = np.concat((query_args["X_start"], X_pool[query_idx]), axis=0)

#             if query_type == "uncertainty":
#                 query_args["X_start"] = learner.X_training
#             elif query_type == "diversity":
#                 query_args["X_start"], _ = get_embeddings(learner.estimator, learner.X_training)

#         if is_stream is False:
#             # remove queried instance from the pool
#             X_pool_org = np.delete(X_pool_org, query_idx, axis=0)
#             # X_pool = np.delete(X_pool, query_idx, axis=0)
#             y_pool = np.delete(y_pool, query_idx)

#             if query_type == "uncertainty":
#                 X_pool = X_pool_org
#             elif query_type == "diversity":
#                 X_pool, _ = get_embeddings(learner.estimator, X_pool_org)

#         else:  # stream based, prepare next batch
#             if X_pool_full.shape[0] < batch_size:
#                 batch_size = X_pool_full.shape[0]
#             batch_idxs = np.random.choice(X_pool_full.shape[0], size=batch_size, replace=False)

#             X_pool, y_pool = X_pool_full[batch_idxs], y_pool_full[batch_idxs]
#             X_pool_full = np.delete(X_pool_full, batch_idxs, axis=0)
#             y_pool_full = np.delete(y_pool_full, batch_idxs, axis=0)

#         # update input dictionary for the query function
#         query_args["X_pool"] = X_pool

#         # update other parameters of query strategy
#         # if "X_start" in query_args:
#         #     # query_args["X_start"] = np.concat((query_args["X_start"], X_pool[query_idx]), axis=0) if "X_pool_emb" in query_args else learner.X_training
#         #     query_args["X_start"] = np.concat((query_args["X_start"], X_pool[query_idx]), axis=0)
#         if "classifier" in query_args:
#             query_args["classifier"] = learner.estimator

#         if kept is True:
#             accuracy = score_accuracy(learner, acc, X_pool_org, y_pool, X_test, y_test)
#             acc_list.append(accuracy)
            
#             y_pred = learner.predict(X_test)
#             f1_scr = f1_score(y_test, y_pred, average="macro")
#             f1_list.append(f1_scr)

#             if goal_metric == "f1":
#                 metric_val = f1_scr
#             elif goal_metric == "acc":
#                 metric_val = accuracy

#             instances_sampled += n_instances
#             inst_list.append(instances_sampled)

#     return acc_list, f1_list, inst_list



def cycle_AL(
    X_train: np.ndarray | None,
    y_train: np.ndarray | None,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifier: Any,
    query_strategy: Callable,
    committee_classifiers: list | None = None,
    n_instances: int = 1,
    batch_size: int = 0,
    start_len: int = 50,
    K: int = 3,
    metric: str = "euclidean",
    input_dim: int = 512,
    output_dim: int = 10,
    max_epochs: int = 50,
    quantile: float = 0.5,
    goal_metric: str = "f1",
    goal_metric_val: float = 0.75,
    acc: str = "test",
    checkpoint_name: str = "",
) -> tuple[list, list]:
    """Executes Active Learning on provided data with the specified strategy and parameters.

    Note:
        If batch_size is specified, then it become stream based batch sampling.
        At every iteration, saves a batch of points, performs the query strategy on this batch, then discards the
        points that are not chosen.

    Inputs:
        X_train: initial labeled data
        y_train: labels for X_train
            both can be "None" if starting without labeled data
        X_pool: initial unlabeled data
        y_pool: labels for X_pool
        X_test: test data
        y_test_ lebels for test data
        classifier: model type used for classification
            supports "randomforest", "nnet" and the other values in ValidClassifiers, or an already instantiated model
        committee_classifiers: list of classifiers to be used for the committee. If None, no committee is used
        query_strategy: one the implemented query strategies imported from the queries folder.
        n_instances: number of instances to be queried at each iteration
        batch_size: if left at 0 is ignored, otherwise the size of the batch for stream based batch sampling
        start_len: for stream based sampling, number of points to be kept as initial set

        K: number of clusters, only if query strategy is k-means
        metric: density metric, only if query strategy is density
        quantile: quantile of distances for stream based diversity query threshold

        input_dim: size of inputs, in case a nnet classifier is passed
        output_dim: number of classes, in case a nnet classifier is passed
        max_epochs: max number of epochs for the training of a nnet at every iteration
        goal_acc: querying continues until this accuracy is reached, or all the pool has been sampled
        acc: whether to asses the classification accuracy on test set, remaining pool, or joint.

    Outputs:
        acc_list: list of accuracy scores, one for each iteration
        inst_list: keeps track of how many instances have been sampled after each iteration
    """

    
    if committee_classifiers is None:
        committee_classifiers = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # check if is stream based batch sampling or not
    if batch_size == 0:
        is_stream = False
    else:
        is_stream = True

    # torch.manual_seed(42)

    # declare learner as ActiveLearner class
    valid_classifiers = [c.value for c in ValidClassifiers]
    if classifier in valid_classifiers:
        learner = set_learner(classifier, query_strategy, X_train, y_train, input_dim, output_dim, max_epochs, device)
    else:
        # in this case the classifier is assumed to be a non-default sklearn classifier
        # if the model is wrongly specified, activelearner will give an error
        # learner = ActiveLearner(
        #     estimator=classifier, query_strategy=query_strategy, X_training=X_train, y_training=y_train
        # )
        learner = SkorchActiveLearner(
            estimator=classifier, query_strategy=query_strategy, X_training=X_train, y_training=y_train
        )

        classifier_copy = copy.deepcopy(classifier)

    
    if committee_classifiers != []:
        committee = set_committee(
            committee_classifiers, query_strategy, X_train, y_train, input_dim, output_dim, max_epochs, device
        )

    if is_stream is True:
        # initialize first batch
        X_pool_full, y_pool_full = X_pool, y_pool
        if X_pool_full.shape[0] < batch_size:
            batch_size = X_pool_full.shape[0]
        batch_idxs = np.random.choice(range(X_pool_full.shape[0]), size=batch_size, replace=False)
        X_pool, y_pool = X_pool_full[batch_idxs], y_pool[batch_idxs]
        X_pool_full = np.delete(X_pool_full, batch_idxs, axis=0)
        y_pool_full = np.delete(y_pool_full, batch_idxs, axis=0)

    # initialize query args dict if empty
    query_args = set_query_args(
        learner.estimator,
        query_strategy,
        X_pool,
        y_pool,
        learner.X_training,
        # X_start,
        n_instances,
        K,
        metric,
        quantile,
        start_len,
    )

    n_instances = query_args["n_instances"]
    if X_train is None:
        instances_sampled = 0
    else:
        instances_sampled = X_train.shape[0]

    # init acc_list and inst_list, and calculate starting values if there's a starting set
    accuracy = 0
    f1_scr = 0
    inst_list = []
    acc_list = []
    f1_list = []
    if X_train is not None:
        accuracy = score_accuracy(learner, acc, X_pool, y_pool, X_test, y_test)
        acc_list.append(accuracy)

        y_pred = learner.predict(X_test)
        f1_scr = f1_score(y_test, y_pred, average="macro")
        f1_list.append(f1_scr)

        inst_list.append(instances_sampled)

    metric_val = 0.0
    if goal_metric == "f1":
        metric_val = f1_scr
    elif goal_metric == "acc":
        metric_val = accuracy
        
    round_counter = 1

    print(f"[INFO] Round {round_counter} - Starting AL cycle with {instances_sampled} instances sampled. Initial {goal_metric}: {metric_val:.4f}")
    # -------------------------------------------------------
    # iterates until goal accuracy is reached or all pool is sampled
    # while accuracy < goal_acc and X_pool.shape[0] > 0:
    while metric_val < goal_metric_val and X_pool.shape[0] > 0:

        if X_pool.shape[0] - n_instances < 0:
            # if there's less instances left, sample all of them
            n_instances = X_pool.shape[0]
            query_args["n_instances"] = n_instances

        stream_queries = [
            stream_query_diversity,
            stream_query_coreset,
            stream_query_probcover,
            stream_query_random,
            stream_query_uncertainty,
            stream_query_margin,
            stream_query_entropy,
        ]
        if query_strategy in stream_queries:  # stream based query
            query_idx, _, kept = query_strategy(**query_args)
        elif committee_classifiers != []:  # committee based query
            query_idx, _ = committee.query(**query_args)
            kept = True
        else:  # pool based query
            query_idx, _ = learner.query(**query_args)
            kept = True
        
        # t-SNE: Generate plot only on rounds 1 and 3
        if kept is True and round_counter in [1, 3]:
            try:
                plot_tsne_selections(
                    classifier=learner.estimator, 
                    X_pool=X_pool, 
                    y_pool=y_pool, 
                    query_idx=query_idx, 
                    strategy_name=query_strategy, 
                    round_num=round_counter
                )
            except Exception as e:
                print(f"[WARNING] t-SNE plotting failed: {e}")

        if kept is True:
            if n_instances == 1:
                shape = n_instances
            else:
                shape = len(query_idx)
            if classifier == "nnet_bo":  # re-instantiate the classifier
                # just initializing doesn't work
                learner.estimator = NeuralNetClassifier(
                    reshapedVGG(num_classes=output_dim),
                    max_epochs=max_epochs,
                    criterion=nn.CrossEntropyLoss,
                    optimizer=torch.optim.Adam,
                    train_split=None,
                    verbose=0,
                    device=device,
                )
            
            learner.estimator = classifier_copy
            classifier_copy = copy.deepcopy(classifier_copy)

            learner.teach(
                X_pool[query_idx],
                y_pool[query_idx],
                only_new=False,
                checkpoint_name=checkpoint_name,
            )  # appends instances to labeled set
            if committee_classifiers != []:
                committee.teach(
                    X_pool[query_idx].reshape(shape, -1),
                    y_pool[query_idx].reshape(
                        shape,
                    ),
                )

        if "X_start" in query_args:
            query_args["X_start"] = learner.X_training
            
        if is_stream is False:
            # remove queried instance from the pool
            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx)

        else:  # stream based, prepare next batch
            if X_pool_full.shape[0] < batch_size:
                batch_size = X_pool_full.shape[0]
            batch_idxs = np.random.choice(X_pool_full.shape[0], size=batch_size, replace=False)

            X_pool, y_pool = X_pool_full[batch_idxs], y_pool_full[batch_idxs]
            X_pool_full = np.delete(X_pool_full, batch_idxs, axis=0)
            y_pool_full = np.delete(y_pool_full, batch_idxs, axis=0)

        # update input dictionary for the query function
        query_args["X_pool"] = X_pool

        # update other parameters of query strategy
        # if "X_start" in query_args:
        #     # query_args["X_start"] = np.concat((query_args["X_start"], X_pool[query_idx]), axis=0) if "X_pool_emb" in query_args else learner.X_training
        if "classifier" in query_args:
            query_args["classifier"] = learner.estimator

        if kept is True:
            accuracy = score_accuracy(learner, acc, X_pool, y_pool, X_test, y_test)
            acc_list.append(accuracy)
            
            y_pred = learner.predict(X_test)
            f1_scr = f1_score(y_test, y_pred, average="macro")
            f1_list.append(f1_scr)

            if goal_metric == "f1":
                metric_val = f1_scr
            elif goal_metric == "acc":
                metric_val = accuracy

            instances_sampled += n_instances
            inst_list.append(instances_sampled)

        round_counter += 1

    return acc_list, f1_list, inst_list



# def strategy_comparison(
#     X_train: np.ndarray | None,  # embedding if representative query strategy otherwise original data
#     y_train: np.ndarray | None,
#     # X_pool: np.ndarray, # embedding if representative query strategy otherwise original data
#     y_pool: np.ndarray,
#     X_pool_org: np.ndarray | None, # original X_pool *** not embeddings!!! ***
#     X_test: np.ndarray,
#     y_test: np.ndarray,
#     classifier: Any,
#     # X_train_org: np.ndarray | None = None, # original X_train / X_initial *** not embeddings!!! ***
#     query_type: Literal["uncertainty", "diversity"],
#     query_strategies: list | None = None,
#     committee_classifiers: list | None = None,
#     n_instances: list | None = None,
#     batch_size: int = 0,
#     start_len: int = 50,
#     K: int = 3,
#     metric: str = "euclidean",
#     quantile: float = 0.5,
#     input_dim: int = 512,
#     output_dim: int = 10,
#     max_epochs: int = 50,
#     # goal_acc: float | None = 0.90,
#     # goal_f1: float | None = 0.70,
#     goal_metric: str = "f1",
#     goal_metric_val: float = 0.75,
#     acc: str = "test",
# ) -> list[pd.DataFrame]:
#     """Executes Active Learning on provided data with many strategies and parameters, and returns a data frame
#     for comparing results.

#     Inputs:
#         X_train: initial labeled data
#         y_train: labels for X_train
#             both can be "None" if starting without labeled data
#         X_pool: initial unlabeled data
#         y_pool: labels for X_pool
#         X_test: test data
#         y_test_ lebels for test data
#         classifier: model type used for classification
#             supports "randomforest", "nnet" and the other values in ValidClassifiers, or an already instantiated model
#         query_strategies: list containing the query strategies to try, from the ones implemented in the
#             queries folder. Default is None, which compares three standard strategies (coreset, uncertainty and random)
#         committee_classifiers: list of classifiers to be used for the committee. If None, no committee is used
#         n_instances: list of parameters for the number of instances to be queried at each iteration.
#             If longer than one, all queries are tried with each parameter.
#         batch_size: if left at 0 is ignored, otherwise the size of the batch for stream based batch sampling
#         start_len: for stream based sampling, number of points to be accepted as initial set

#         K: number of clusters, only if one of the query strategies is k-means
#         metric: density metric, only if one of the query strategies is density
#         quantile: quantile of distances for stream based diversity query threshold
#         input_dim: size of inputs, in case a nnet classifier is passed
#         output_dim: number of classes, in case a nnet classifier is passed
#         max_epochs: max number of epochs for the training of a nnet at every iteration

#         goal_acc: querying continues until this accuracy is reached, or all the pool has been sampled
#         acc: whether to asses the classification accuracy on test set, remaining pool, or joint.

#     Outputs:
#         scores_list: list of data frames containing the results, accuracy and number of instances for each
#             query strategy. Number of dataframes corresponds to number of parameters to try specified
#             in n_instances
#     """
#     # set default values for None-arguments
#     if query_strategies is None:
#         query_strategies = [query_coreset, uncertainty_sampling, query_random]
#     if committee_classifiers is None:
#         committee_classifiers = []
#     if n_instances is None:
#         n_instances = [1]

#     # column names for output data frame
#     col_names = set_query_names(query_strategies)
#     # duplicate columns: one for accuracy, one for instances (update:one for f1)
#     # col_names = [item for item in col_names for _ in range(2)] * len(n_instances)
#     col_names = [item for item in col_names for _ in range(3)] * len(n_instances)

#     i = 1  # counter combinations
#     j = 0  # counter columns

#     scores_list = []

#     for n_inst in n_instances:
#         scores = pd.DataFrame()
#         for query in query_strategies:
#             logging.info(f"---computing combination {i}")
#             accs, f1s, insts = cycle_AL(
#                 X_train=X_train,
#                 y_train=y_train,
#                 # X_pool=X_pool,
#                 y_pool=y_pool,
#                 X_test=X_test,
#                 y_test=y_test,
#                 X_pool_org=X_pool_org,
#                 # X_train_org=X_train_org,
#                 classifier=classifier,
#                 query_type = query_type,
#                 query_strategy=query,
#                 committee_classifiers=committee_classifiers,
#                 n_instances=n_inst,
#                 batch_size=batch_size,
#                 start_len=start_len,
#                 K=K,
#                 metric=metric,
#                 quantile=quantile,
#                 input_dim=input_dim,
#                 output_dim=output_dim,
#                 max_epochs=max_epochs,
#                 goal_metric=goal_metric,
#                 goal_metric_val=goal_metric_val,
#                 acc=acc,
#             )

#             # col_names[j + 1] += " inst"
#             col_names[j] += " accuracy"
#             col_names[j + 1] += " f1 score"
#             col_names[j + 2] += " inst"

#             # df_temp = pd.DataFrame({col_names[j]: accs, col_names[j + 1]: insts})
#             # scores = pd.concat([scores, df_temp], axis=1)
#             df_temp = pd.DataFrame({col_names[j]: accs, col_names[j + 1]: f1s, col_names[j + 2]: insts})
#             scores = pd.concat([scores, df_temp], axis=1)

#             i += 1
#             # j += 2
#             j += 3
#         scores_list.append(scores)
#     logging.info("Done.")

#     return scores_list




def strategy_comparison(
    X_train: np.ndarray | None,  # embedding if representative query strategy otherwise original data
    y_train: np.ndarray | None,
    X_pool: np.ndarray, # embedding if representative query strategy otherwise original data
    y_pool: np.ndarray,
    # X_pool_org: np.ndarray | None, # original X_pool *** not embeddings!!! ***
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifier: Any,
    # X_train_org: np.ndarray | None = None, # original X_train / X_initial *** not embeddings!!! ***
    # query_type: Literal["uncertainty", "diversity"],
    query_strategies: list | None = None,
    committee_classifiers: list | None = None,
    n_instances: list | None = None,
    batch_size: int = 0,
    start_len: int = 50,
    K: int = 3,
    metric: str = "euclidean",
    quantile: float = 0.5,
    input_dim: int = 512,
    output_dim: int = 10,
    max_epochs: int = 50,
    # goal_acc: float | None = 0.90,
    # goal_f1: float | None = 0.70,
    goal_metric: str = "f1",
    goal_metric_val: float = 0.75,
    acc: str = "test",
    checkpoint_name: str = None,
) -> list[pd.DataFrame]:
    """Executes Active Learning on provided data with many strategies and parameters, and returns a data frame
    for comparing results.

    Inputs:
        X_train: initial labeled data
        y_train: labels for X_train
            both can be "None" if starting without labeled data
        X_pool: initial unlabeled data
        y_pool: labels for X_pool
        X_test: test data
        y_test_ lebels for test data
        classifier: model type used for classification
            supports "randomforest", "nnet" and the other values in ValidClassifiers, or an already instantiated model
        query_strategies: list containing the query strategies to try, from the ones implemented in the
            queries folder. Default is None, which compares three standard strategies (coreset, uncertainty and random)
        committee_classifiers: list of classifiers to be used for the committee. If None, no committee is used
        n_instances: list of parameters for the number of instances to be queried at each iteration.
            If longer than one, all queries are tried with each parameter.
        batch_size: if left at 0 is ignored, otherwise the size of the batch for stream based batch sampling
        start_len: for stream based sampling, number of points to be accepted as initial set

        K: number of clusters, only if one of the query strategies is k-means
        metric: density metric, only if one of the query strategies is density
        quantile: quantile of distances for stream based diversity query threshold
        input_dim: size of inputs, in case a nnet classifier is passed
        output_dim: number of classes, in case a nnet classifier is passed
        max_epochs: max number of epochs for the training of a nnet at every iteration

        goal_acc: querying continues until this accuracy is reached, or all the pool has been sampled
        acc: whether to asses the classification accuracy on test set, remaining pool, or joint.

    Outputs:
        scores_list: list of data frames containing the results, accuracy and number of instances for each
            query strategy. Number of dataframes corresponds to number of parameters to try specified
            in n_instances
    """
    # set default values for None-arguments
    if query_strategies is None:
        query_strategies = [query_coreset, uncertainty_sampling, query_random]
    if committee_classifiers is None:
        committee_classifiers = []
    if n_instances is None:
        n_instances = [1]

    # column names for output data frame
    col_names = set_query_names(query_strategies)
    # duplicate columns: one for accuracy, one for instances (update:one for f1)
    # col_names = [item for item in col_names for _ in range(2)] * len(n_instances)
    col_names = [item for item in col_names for _ in range(3)] * len(n_instances)

    i = 1  # counter combinations
    j = 0  # counter columns

    scores_list = []

    for n_inst in n_instances:
        scores = pd.DataFrame()
        for query in query_strategies:
            logging.info(f"---computing combination {i}")
            accs, f1s, insts = cycle_AL(
                X_train=X_train,
                y_train=y_train,
                X_pool=X_pool,
                y_pool=y_pool,
                X_test=X_test,
                y_test=y_test,
                classifier=classifier,
                query_strategy=query,
                committee_classifiers=committee_classifiers,
                n_instances=n_inst,
                batch_size=batch_size,
                start_len=start_len,
                K=K,
                metric=metric,
                quantile=quantile,
                input_dim=input_dim,
                output_dim=output_dim,
                max_epochs=max_epochs,
                goal_metric=goal_metric,
                goal_metric_val=goal_metric_val,
                acc=acc,
                checkpoint_name=checkpoint_name,
            )

            # col_names[j + 1] += " inst"
            col_names[j] += " accuracy"
            col_names[j + 1] += " f1 score"
            col_names[j + 2] += " inst"

            # df_temp = pd.DataFrame({col_names[j]: accs, col_names[j + 1]: insts})
            # scores = pd.concat([scores, df_temp], axis=1)
            df_temp = pd.DataFrame({col_names[j]: accs, col_names[j + 1]: f1s, col_names[j + 2]: insts})
            scores = pd.concat([scores, df_temp], axis=1)

            i += 1
            # j += 2
            j += 3
        scores_list.append(scores)
    logging.info("Done.")

    return scores_list






# def plot_results(
#     scores_list: list, n_instances: list, tot_samples: int, figsize: tuple = (13, 8), goal_acc: int = 0.9
# ) -> None:
#     """Plots scores from strategy_comparison.

#     Inputs:
#         scores_list: output from strategy_comparison. List of dataframes containing scores
#         n_instances: list of parameters used for n_instances. Determines number of plots, one for each parameter
#         tot_samples: number of samples in original data, for scale in the graph
#         figsize: optional, tuple determining graph size
#         goal_acc: accuracy of the full data classifier, displays a yellow line

#     Outputs:
#         One plot for each parameter choice of n_instances. X axis is number of instances sampled, Y axis is accuracy.
#         One line on the plot for each query strategy
#     """
#     ngraphs = len(scores_list)
#     fig, axarr = plt.subplots(ngraphs, 1, figsize=figsize)

#     for i in range(ngraphs):
#         scores = scores_list[i]
#         row = i
#         for j in range(0, scores.shape[1], 2):
#             if ngraphs == 1:
#                 plot_j = axarr
#             else:
#                 plot_j = axarr[row]
#             plot_j.plot(scores.iloc[:, j + 1], scores.iloc[:, j], label=scores.columns[j])
#             plot_j.legend()
#             plot_j.grid(True)
#             plot_j.set_title(f"instances per iter: {n_instances[i]}", fontsize=10)
#             x_ticks = range(0, tot_samples, n_instances[i])
#             plot_j.set_xticks(x_ticks)
#             plot_j.axhline(y=goal_acc, color="y", linestyle="--")  # goal acc
#             plot_j.set_xlabel("instances")
#             plot_j.set_ylabel("accuracy")

#     fig.tight_layout()
#     fig.suptitle(f"Accuracy of different strategies until {goal_acc} accuracy is reached", y=1.02)

def plot_results(
    scores_list: list, n_instances: list, tot_samples: int, figsize: tuple = (13, 8), goal_metric="f1", goal_metric_val: int = 0.9, save_path: str = None
) -> None:
    """Plots scores from strategy_comparison.

    Inputs:
        scores_list: output from strategy_comparison. List of dataframes containing scores
        n_instances: list of parameters used for n_instances. Determines number of plots, one for each parameter
        tot_samples: number of samples in original data, for scale in the graph
        figsize: optional, tuple determining graph size
        goal_acc: accuracy of the full data classifier, displays a yellow line

    Outputs:
        One plot for each parameter choice of n_instances. X axis is number of instances sampled, Y axis is accuracy.
        One line on the plot for each query strategy
    """
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.grid(True, linestyle='--', alpha=0.6)

    ngraphs = len(scores_list)
    fig, axarr = plt.subplots(ngraphs, 1, figsize=figsize, sharex=False)

    # Ensure axarr is always a list for consistency
    if ngraphs == 1:
        axarr = [axarr]

    # Color palette for distinct lines
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i in range(ngraphs):
        scores = scores_list[i]
        ax = axarr[i]
        
        # Determine strict y-axis limits to make differences visible
        # (Optional: remove set_ylim if you want auto-scaling)
        y_min, y_max = 1.0, 0.0

        # Loop through strategies (assumes structure: [Acc, F1, Inst, Acc, F1, Inst...])
        strategy_idx = 0
        for j in range(0, scores.shape[1], 3):
            
            # --- Extract Data ---
            x_data = scores.iloc[:, j + 2] # The 'instances' column
            
            if goal_metric == "acc":
                y_data = scores.iloc[:, j]
                raw_label = scores.columns[j]
                ylabel = "Accuracy"
            elif goal_metric == "f1":
                y_data = scores.iloc[:, j+1]
                raw_label = scores.columns[j+1]
                ylabel = "Macro F1 Score"

            # --- Clean Label ---
            # Removes " f1 score" or " accuracy" to keep legend clean
            clean_label = raw_label.replace(" f1 score", "").replace(" accuracy", "").replace("inst", "").strip()
            
            # --- Plotting ---
            color = colors[strategy_idx % len(colors)]
            ax.plot(
                x_data, 
                y_data, 
                label=clean_label, 
                marker='o',       # Add dots
                markersize=5, 
                linewidth=2, 
                linestyle='-',
                alpha=0.8,
                color=color
            )
            
            # Track min/max for scaling
            y_min = min(y_min, y_data.min())
            y_max = max(y_max, y_data.max())
            strategy_idx += 1

        # --- Goal Line (Red Dashed) ---
        ax.axhline(y=goal_metric_val, color="#d62728", linestyle="--", linewidth=1.5, label="Target Goal")
        # Add text annotation for the goal
        ax.text(x_data.min(), goal_metric_val + 0.005, f" Goal: {goal_metric_val:.3f}", 
                color="#d62728", fontsize=9, fontweight='bold', va='bottom')

        # --- Formatting ---
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_xlabel("Labeled Instances Count", fontsize=11)
        ax.set_title(f"Active Learning Performance (Batch Size: {n_instances[i]})", fontsize=13)
        
        ax.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add a bit of padding to Y-axis range
        ax.set_ylim(y_min - 0.02, max(y_max, goal_metric_val) + 0.02)

    plt.suptitle(f"Strategy Comparison: {ylabel}", y=1.02, fontsize=16)
    fig.tight_layout()

    # --- Save Logic ---
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Plot saved to: {save_path}")
    
    # plt.show()