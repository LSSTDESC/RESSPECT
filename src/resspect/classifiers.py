# Copyright 2020 resspect software
# Author: The RESSPECT team
#
# created on 14 April 2020
#
# Licensed GNU General Public License v3.0;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from sklearn.ensemble import RandomForestClassifier
#from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample
from sklearn.utils.validation import check_is_fitted

import os
import networkx as nx
import tf_keras

__all__ = ['random_forest',#'gradient_boosted_trees',
           'knn',
           'mlp','svm','nbg', 'bootstrap_clf'
          ]


def bootstrap_clf(clf_function, n_ensembles, train_features,
                  train_labels, test_features, **kwargs):
    """
    Train an ensemble of classifiers using bootstrap.

    Parameters
    ----------
    clf_function: function
        function to train classifier
    n_ensembles: int
        number of classifiers in the ensemble
    train_features: np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Test sample features.
    kwargs: extra parameters
        All keywords required by
        sklearn.ensemble.RandomForestClassifier function.

    Returns
    -------
    predictions: np.array
        Prediction of the ensemble
    class_prob: np.array
        Average distribution of ensemble members
    ensemble_probs: np.array
        Probability output of each member of the ensemble
    """
    n_labels = np.unique(train_labels).size
    num_test_data = test_features.shape[0]
    ensemble_probs = np.zeros((num_test_data, n_ensembles, n_labels))
    classifier_list = list()
    for i in range(n_ensembles):
        x_train, y_train = resample(train_features, train_labels)
        predicted_class, class_prob, clf = clf_function(x_train,
                                                        y_train,
                                                        test_features,
                                                        **kwargs)
        #clf = clf_function(**kwargs)
        #clf.fit(x_train, y_train)
        #predicted_class = clf.predict(test_features)
        #class_prob = clf.predict_proba(test_features)
        
        classifier_list.append((str(i), clf))
        ensemble_probs[:, i, :] = class_prob

    ensemble_clf = PreFitVotingClassifier(classifier_list, voting='soft')  #Must use soft voting
    class_prob = ensemble_probs.mean(axis=1)
    predictions = np.argmax(class_prob, axis=1)
    
    return predictions, class_prob, ensemble_probs, ensemble_clf


def random_forest(train_features:  np.array, train_labels: np.array,
                  test_features: np.array, n_estimators=1000, retrain: bool = True, curr_clf=None, **kwargs):
    """Random Forest classifier.

    Parameters
    ----------
    train_features: np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Features from sample to be classified.
    n_estimators: int (optional)
        Number of trees in the forest. Default is 1000.
    kwargs: extra parameters
        All keywords required by
        sklearn.ensemble.RandomForestClassifier function.

    Returns
    -------
    predictions: np.array
        Predicted classes for test sample.
    prob: np.array
        Classification probability for test sample [pIa, pnon-Ia].
    """

    # create classifier instance
    clf = RandomForestClassifier(n_estimators=n_estimators, **kwargs) if retrain == True else curr_clf
    clf.fit(train_features, train_labels)                     # train
    predictions = clf.predict(test_features)                # predict
    prob = clf.predict_proba(test_features)       # get probabilities

    return predictions, prob, clf
  
#######################################################################
######  we need to find a non-bugged version of xgboost ##############

#def gradient_boosted_trees(train_features: np.array,
#                           train_labels: np.array,
#                           test_features: np.array, **kwargs):
    """Gradient Boosted Trees classifier.

    Parameters
    ----------
    train_features : np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Test sample features.
    kwargs: extra parameters
        All parameters allowed by sklearn.XGBClassifier

    Returns
    -------
    predictions: np.array
        Predicted classes.
    prob: np.array
        Classification probability for all objects, [pIa, pnon-Ia].
    """

    #create classifier instance
#    clf = XGBClassifier(**kwargs)

#    clf.fit(train_features, train_labels)             # train
#    predictions = clf.predict(test_features)          # predict
#    prob = clf.predict_proba(test_features)           # get probabilities

#    return predictions, prob, clf
#########################################################################

def knn(train_features: np.array, train_labels: np.array,
        test_features: np.array, retrain: bool = True, curr_clf=None, **kwargs):

    """K-Nearest Neighbour classifier.

    Parameters
    ----------
    train_features : np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Test sample features.
    kwargs: extra parameters
        All parameters allowed by sklearn.neighbors.KNeighborsClassifier

    Returns
    -------
    predictions: np.array
        Predicted classes.
    prob: np.array
        Classification probability for all objects, [pIa, pnon-Ia].
    """

    #create classifier instance
    clf = KNeighborsClassifier(**kwargs) if retrain == True else curr_clf

    clf.fit(train_features, train_labels)              # train
    predictions = clf.predict(test_features)           # predict
    prob = clf.predict_proba(test_features)            # get probabilities

    return predictions, prob, clf


def mlp(train_features: np.array, train_labels: np.array,
        test_features: np.array, retrain: bool = True, curr_clf=None, **kwargs):

    """Multi Layer Perceptron classifier.

    Parameters
    ----------
    train_features : np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Test sample features.
    kwargs: extra parameters
        All parameters allowed by sklearn.neural_network.MLPClassifier

    Returns
    -------
    predictions: np.array
        Predicted classes.
    prob: np.array
        Classification probability for all objects, [pIa, pnon-Ia].
    """

    #create classifier instance
    clf = MLPClassifier(**kwargs) if retrain == True else curr_clf

    clf.fit(train_features, train_labels)              # train
    predictions = clf.predict(test_features)           # predict
    prob = clf.predict_proba(test_features)            # get probabilities

    return predictions, prob, clf

def svm(train_features: np.array, train_labels: np.array,
        test_features: np.array, retrain: bool = True, curr_clf=None, **kwargs):
    """Support Vector classifier.

    Parameters
    ----------
    train_features : np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Test sample features.
    kwargs: dict (optional)
        All parameters which can be passed to sklearn.svm.SVC
        function.

    Returns
    -------
    predictions: np.array
        Predicted classes.
    prob: np.array
        Classification probability for all objects, [pIa, pnon-Ia].
    """

    #create classifier instance
    clf = SVC(probability=True, **kwargs) if retrain == True else curr_clf

    clf.fit(train_features, train_labels)          # train
    predictions = clf.predict(test_features)       # predict
    prob = clf.predict_proba(test_features)        # get probabilities

    return predictions, prob, clf
  

def nbg(train_features: np.array, train_labels: np.array,
                  test_features: np.array, retrain: bool = True, curr_clf=None, **kwargs):

    """Naive Bayes classifier.

    Parameters
    ----------
    train_features : np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Test sample features.
    kwargs: dict (optional)
        All parameters which can be passed to sklearn.svm.SVC
        function.

    Returns
    -------
    predictions: np.array
        Predicted classes.
    prob: np.array
        Classification probability for all objects, [pIa, pnon-Ia].
    """

    #create classifier instance
    clf=GaussianNB(**kwargs) if retrain == True else curr_clf

    clf.fit(train_features, train_labels)         # fit
    predictions = clf.predict(test_features)      # predict
    prob = clf.predict_proba(test_features)       # get probabilities

    return predictions, prob, clf

class PreFitVotingClassifier(object):
    """Stripped-down version of VotingClassifier that uses prefit estimators"""
    def __init__(self, estimators, voting='hard', weights=None):
        self.estimators = [e[1] for e in estimators]
        self.named_estimators = dict(estimators)
        self.voting = voting
        self.weights = weights

    def fit(self, X, y, sample_weight=None):
        raise NotImplementedError

    def predict(self, X):
        """ Predict class labels for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        """

        check_is_fitted(self, 'estimators')
        if self.voting == 'soft':
            maj = np.argmax(self.predict_proba(X), axis=1)

        else:  # 'hard' voting
            predictions = self._predict(X)
            maj = np.apply_along_axis(lambda x:
                                      np.argmax(np.bincount(x,
                                                weights=self.weights)),
                                      axis=1,
                                      arr=predictions.astype('int'))
        return maj

    def _collect_probas(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict_proba(X) for clf in self.estimators])

    def _predict_proba(self, X):
        """Predict class probabilities for X in 'soft' voting """
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available when"
                                 " voting=%r" % self.voting)
        check_is_fitted(self, 'estimators')
        avg = np.average(self._collect_probas(X), axis=0, weights=self.weights)
        return avg

    @property
    def predict_proba(self):
        """Compute probabilities of possible outcomes for samples in X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        return self._predict_proba

    def transform(self, X):
        """Return class labels or probabilities for X for each estimator.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        -------
        If `voting='soft'`:
          array-like = [n_classifiers, n_samples, n_classes]
            Class probabilities calculated by each classifier.
        If `voting='hard'`:
          array-like = [n_samples, n_classifiers]
            Class labels predicted by each classifier.
        """
        check_is_fitted(self, 'estimators')
        if self.voting == 'soft':
            return self._collect_probas(X)
        else:
            return self._predict(X)

    def _predict(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.estimators]).T

class ORACLE:
    def __init__(self):
        os.environ["TF_USE_LEGACY_KERAS"] = 1
        self.source_node_label = "Alert"
        self.model = None
        self.tree = None
        self.y_pred = None
        self.class_probabilities = None
        self.class_conditional_probabilities = None
        self.true_labels = []
        self.pred_labels = []
        
        self.get_taxonomy_tree()
    
    def get_taxonomy_tree(self):
        # Graph to store taxonomy
        self.tree = nx.DiGraph(directed=True)

        self.tree.add_node('Alert', color='red')

        # Level 1
        level_1_nodes = ['Transient', 'Variable']
        self.tree.add_nodes_from(level_1_nodes)
        self.tree.add_edges_from([('Alert', level_1_node) for level_1_node in level_1_nodes])

        # Level 2a nodes for Transients
        level_2a_nodes = ['SN', 'Fast', 'Long']
        self.tree.add_nodes_from(level_2a_nodes)
        self.tree.add_edges_from([('Transient', level_2a_node) for level_2a_node in level_2a_nodes])

        # Level 2b nodes for Transients
        level_2b_nodes = ['Periodic', 'AGN']
        self.tree.add_nodes_from(level_2b_nodes)
        self.tree.add_edges_from([('Variable', level_2b_node) for level_2b_node in level_2b_nodes])

        # Level 3a nodes for SN Transients
        level_3a_nodes = ['SNIa', 'SNIb/c', 'SNIax', 'SNI91bg', 'SNII']
        self.tree.add_nodes_from(level_3a_nodes)
        self.tree.add_edges_from([('SN', level_3a_node) for level_3a_node in level_3a_nodes])

        # Level 3b nodes for Fast events Transients
        level_3b_nodes = ['KN', 'Dwarf Novae', 'uLens', 'M-dwarf Flare']
        self.tree.add_nodes_from(level_3b_nodes)
        self.tree.add_edges_from([('Fast', level_3b_node) for level_3b_node in level_3b_nodes])

        # Level 3c nodes for Long events Transients
        level_3c_nodes = ['SLSN', 'TDE', 'ILOT', 'CART', 'PISN']
        self.tree.add_nodes_from(level_3c_nodes)
        self.tree.add_edges_from([('Long', level_3c_node) for level_3c_node in level_3c_nodes])

        # Level 3d nodes for periodic stellar events
        level_3d_nodes = ['Cepheid', 'RR Lyrae', 'Delta Scuti', 'EB'] 
        self.tree.add_nodes_from(level_3d_nodes)
        self.tree.add_edges_from([('Periodic', level_3d_node) for level_3d_node in level_3d_nodes])
        
        self.level_order_nodes = list(nx.bfs_tree(self.tree, source=self.source_node_label).nodes())

    def get_predictions(self, data, path_to_weights=None):
        if path_to_weights is not None:
            self.model = tf_keras.models.load_model(path_to_weights, compile=False)
        else:
            pass # needs to be implemented at some point in the future
        
        self.y_pred = self.model.predict(data)
    
    def get_conditional_probabilities(self):
        # Create a new arrays to store pseudo (conditional) probabilities.
        self.class_conditional_probabilities = np.copy(self.y_pred)
        self.class_probabilities = np.copy(self.y_pred)

        parents = [list(self.tree.predecessors(node)) for node in self.level_order_nodes]
        for idx in range(len(parents)):

            # Make sure the graph is a tree.
            assert len(parents[idx]) == 0 or len(parents[idx]) == 1, 'Number of parents for each node should be 0 (for root) or 1.'
            
            if len(parents[idx]) == 0:
                parents[idx] = ''
            else:
                parents[idx] = parents[idx][0]

        # Finding unique parents for masking
        unique_parents = list(set(parents))
        unique_parents.sort()

        # Create masks for applying soft max and calculating loss values.
        masks = []
        for parent in unique_parents:
            masks.append(self.get_indices_where(parents, parent))
        
        # Get the masked softmaxes
        for mask in masks:
            self.class_probabilities[:, mask] = np.exp(self.y_pred[:, mask]) / np.sum(np.exp(self.y_pred[:, mask]), axis=1, keepdims=True)
            

        for node in self.level_order_nodes:
            
            # Find path from node to 
            path = list(nx.shortest_path(self.tree, self.source_node_label, node))
            
            # Get the index of the node for which we are calculating the pseudo probability
            node_index = self.level_order_nodes.index(node)
            
            # Indices of all the classes in the path from source to the node for which we are calculating the pseudo probability
            path_indices = [self.level_order_nodes.index(u) for u in path]
            
            #print(node, path, node_index, path_indices, pseudo_probabilities[:, path_indices])
            
            # Multiply the pseudo probabilites of all the classes in the path so that we get the conditional pseudo probabilites
            self.class_conditional_probabilities[:, node_index] = np.prod(self.class_probabilities[:, path_indices], axis = 1)
            
    def get_indices_where(arr, target):
        to_return = []
        for i, obs in enumerate(arr):
            if obs == target:
                to_return.append(i)
                
        return to_return

    def sort_nodes_and_generate_labels(self, label_pred, label_true):
        depths = [len(nx.shortest_path(self.tree, source=self.source_node_label, target=node)) for node in self.level_order_nodes]
        
        masks = []
        unique_depths = list(set(depths)).sort()
        
        for depth in unique_depths:
            masks.append(self.get_indices_where(depths, depth))
        
        self.true_labels = []
        self.pred_labels = []
        
        for mask, depth in zip(masks, unique_depths):
            if depth != 0 and depth != 3:
                mask_classes = [self.level_order_nodes[m] for m in mask]
                
                for i in range(label_true.shape[0]):
                    true_class_idx = np.argmax(label_true[i, mask])
                    self.true_labels.append(mask_classes[true_class_idx])
                    
                    pred_class_idx = np.argmax(label_pred[i, mask])
                    self.pred_labels.append(mask_classes[pred_class_idx])

def main():
    return None


if __name__ == '__main__':
    main()
