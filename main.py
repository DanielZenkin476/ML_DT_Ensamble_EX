# This is an exercise project for DT and their ensambles, using iris flower dataset and california housing dataset
# goal - classify type of flower from 4 inputs : sepal_length,sepal_width,petal_length,petal_width
# importing libraries
import numpy as np
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Random seed
seed = 100
np.random.seed(seed)
#get the data
iris = datasets.load_iris()
data = pd.DataFrame({'sepal_length': iris.data[:,0],
                   'sepal_width': iris.data[:,1],
                   'petal_length': iris.data[:,2],
                   'petal_width': iris.data[:,3],
                   'type': iris.target})
print(data)
#split using sklearn

X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values.reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2) # random_state for reproducible output
print(f'x_tr: {X_train.shape} ; y_tr: {Y_train.shape} ; x_tst: {X_test.shape} ; y_tst: {Y_test.shape}')#print shape of splits

# now to create the model - will start from creating a node class:

class Node():
    def __init__(self,feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor '''
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # for leaf node
        self.value = value

#DT class :
class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        ''' constructor '''

        # initialize the root of the tree
        self.names = ['Iris setosa','Iris versicolor','Iris virginica']
        self.root = None

        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree '''

        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)

        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["info_gain"] > 0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["info_gain"])

        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''

        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")

        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)#removes duplicates
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if children are not null
                if len(dataset_left)>0 and len(dataset_right) != 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y)
                    # update the best split if needed
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        # return best split
        return best_split

    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''

        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right

    def information_gain(self, parent, l_child, r_child):
        ''' function to compute information gain '''

        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)

        gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain

    def entropy(self, y):
        ''' function to compute entropy '''

        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''

        Y = list(Y)
        return max(Y, key=Y.count) # the most common label

    def fit(self, X, Y):
        ''' function to train the tree '''

        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        ''' function to predict new dataset '''

        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions

    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''

        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

    def print_tree(self, tree=None, indent=" -"):
      ''' function to print the tree '''

      if not tree:
          tree = self.root

      if tree.value is not None:
          # print(self.names[int(tree.value)])
          print(round(tree.value))

      else:
          print(f'feature{tree.feature_index} <= {tree.threshold}? [info gain: {tree.info_gain:.2f}]')
          print("%sleft:" % (indent), end="")
          self.print_tree(tree.left, " " + indent + "-")
          print("%sright:" % (indent), end="")
          self.print_tree(tree.right, " " + indent + "-")

#create classifier and fit data
classifier = DecisionTreeClassifier(min_samples_split=4,max_depth=4)
classifier.fit(X_train,Y_train)
#print tree
classifier.print_tree()
# plot the data in a heatmap
sns.pairplot(data,hue='type', diag_kind='hist') # Plot pairwise relationships in a dataset.
plt.show()
#calc prediction and accuracy
Y_pred = np.array(classifier.predict(X_test))
print(f'Decision tree model accuracy: {accuracy_score(Y_test, Y_pred)}')
#heatmap to show prediction
mat = confusion_matrix(Y_test, Y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

# now , usuing california housing data set (prices problem ) :

data = pd.read_csv(filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")

data["median_house_value"] /= 1000.0# decrease by 1K factor for ease of use, need to remember to add if showing prices
print(data)
#train/test splits:
indexes = np.random.choice(len(data), size = 1000, replace = False ) # gives 1000 different item indexes from data
X = data.iloc[indexes, :-1].values
Y = data.iloc[indexes, -1].values.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)
print(f'x_tr: {X_train.shape} ; y_tr: {Y_train.shape} ; x_tst: {X_test.shape} ; y_tst: {Y_test.shape}')