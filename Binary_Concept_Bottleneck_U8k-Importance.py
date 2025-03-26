import copy
import warnings

import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax
import argparse

parser = argparse.ArgumentParser(
        prog="FBN",
        description="This program Runs.",
    )
parser.add_argument("-s", "--seed", default=0, type=int)
args = parser.parse_args()


warnings.filterwarnings('ignore')

DATASET_PATH = "Urban8K200.csv"
seed  = args.seed
number_of_classes = 10
number_of_important_features = 5

DATASET = pd.read_csv(DATASET_PATH)

DATASET = DATASET.drop_duplicates( keep='first', inplace=False, ignore_index=False)
DATASET = DATASET.dropna(axis=1,how='any')
DATASET.columns = map(str.lower, DATASET.columns)
# DATASET = DATASET.drop(labels=['filename','length'],axis=1)
feature_names = [feature for feature in DATASET.columns if feature != 'label']
from sklearn.preprocessing import LabelEncoder


encoder = LabelEncoder()
labels = DATASET.iloc[:, -1] # get labels column
labels.to_frame()
y = encoder.fit_transform(labels)
DATASET.iloc[:, -1] = y

TRAIN_DATASET, TEST_DATASET, _, _ = train_test_split(
    DATASET, DATASET["label"], test_size=0.3, stratify=DATASET["label"], random_state=seed
    )
TRAIN_DATASET.shape, TEST_DATASET.shape

training_dataset = copy.deepcopy(TRAIN_DATASET)
testing_dataset = copy.deepcopy(TEST_DATASET)


Important_Features_For_Class = {}
Important_Features_For_Class_value = {}
random_state = args.seed
accs = 0
for label in range(number_of_classes):
    Important_Features_For_Class[label] = []
    # Get Labels

    labels = training_dataset["label"]
    print(labels.shape)

    Features_Of_Label_To_Identify = training_dataset[training_dataset["label"] == label]
    print(Features_Of_Label_To_Identify.shape)
    # Set Label to Zero for sanity!
    Features_Of_Label_To_Identify["label"] = 0
    # Get Features not of Label to Identify (Other classes) .
    Features_Of_Other_Labels =  training_dataset[training_dataset["label"] != label]
    # Sample from here to get a uniform sample
    Features_Of_Other_Labels = Features_Of_Other_Labels.sample(Features_Of_Label_To_Identify.shape[0], random_state=random_state)
    # Set Labels to 1 (So the other features all have the same Label)
    Features_Of_Other_Labels["label"] = 1

    # Create Dataset
    Binary_Classification_Dataset = np.vstack((Features_Of_Label_To_Identify, Features_Of_Other_Labels))
    # Get Features
    Binary_Classification_Dataset_Features = Binary_Classification_Dataset[:,:-1]
    Binary_Classification_Dataset_Features = pd.DataFrame(Binary_Classification_Dataset_Features)
    Binary_Classification_Dataset_Features.columns = range(Binary_Classification_Dataset_Features.columns.size)
    # Get Labels
    Binary_Classification_Dataset_Labels = Binary_Classification_Dataset[:,-1]

    # Create Training and Test Split (Remember we are still only using the original training data)
    X_train, X_test, Y_train, Y_test = train_test_split(
    Binary_Classification_Dataset_Features, Binary_Classification_Dataset_Labels, test_size=0.3, stratify=Binary_Classification_Dataset_Labels, random_state=random_state
    )

    # Normalise Training and Test Data
    X_train = (X_train-X_train.mean())/X_train.std()
    X_test = (X_test-X_test.mean())/X_test.std()

    # Create Binary Classifier and train
    Binary_XGB_Classifier = xgb.XGBClassifier(objective='binary:logistic', random_state=random_state)
    Binary_XGB_Classifier.fit(X_train, Y_train)

    # Evaluate Classifier
    predictions = Binary_XGB_Classifier.predict(X_test)
    xgb_accuracy = accuracy_score(predictions, Y_test)
    accs+=xgb_accuracy
    xgb_f1_score = f1_score(Y_test, predictions, average='weighted')
    print("CLASS: ", label)
    print("Classifier F1 Score:", xgb_f1_score)
    print("Classifier Accuracy:", xgb_accuracy)
    print("Features with Zero Importance: ", np.count_nonzero(Binary_XGB_Classifier.feature_importances_== 0))
    # Get Feature Importance and Allocate to 
    importance = pd.DataFrame(Binary_XGB_Classifier.feature_importances_)
    importance.sort_values(by=0,ignore_index=False,ascending=True,inplace=True)
    importance = importance.iloc[::-1]
    Important_Features_For_Class[label] = list(training_dataset.keys()[importance.index.values])
    Important_Features_For_Class_value[label] = importance.values

uniques = []
number_of_features = np.arange(1,len(feature_names)+1)
for important_features in number_of_features:
    unique_features = []
    for label in range(10):
        for feature in Important_Features_For_Class[label][:important_features]:
            if feature not in unique_features:
                unique_features.append(feature)
    uniques.append(len(unique_features))


class_one_hot_features = {}
class_zero_features = {}
for label in range(number_of_classes):
    class_one_hot_features[label] = []
    for feature in Important_Features_For_Class[label][:number_of_important_features]:
        class_one_hot_features[label].append(feature)
for label in range(number_of_classes):
    class_zero_features[label] = []
    for feature in feature_names:
        if feature not in class_one_hot_features[label]:
            class_zero_features[label].append(feature)
#

for key in class_one_hot_features:
    print(key, class_one_hot_features[key])

OneHotEncodedDataset = copy.deepcopy(training_dataset)
OneHotEncodedDataset_test = copy.deepcopy(testing_dataset)
for label in range(number_of_classes):
    Important_Features_For_Class_value[label][:number_of_important_features] = softmax(Important_Features_For_Class_value[label][:number_of_important_features], axis=0)

    OneHotEncodedDataset.loc[OneHotEncodedDataset['label'].eq(label),class_one_hot_features[label]] = Important_Features_For_Class_value[label][:number_of_important_features].T
    OneHotEncodedDataset.loc[OneHotEncodedDataset['label'].eq(label),class_zero_features[label]]=0
    OneHotEncodedDataset_test.loc[OneHotEncodedDataset_test['label'].eq(label),class_one_hot_features[label]] = Important_Features_For_Class_value[label][:number_of_important_features].T
    OneHotEncodedDataset_test.loc[OneHotEncodedDataset_test['label'].eq(label),class_zero_features[label]]=0


important_features = list(set().union(*class_one_hot_features.values()))[:number_of_important_features]

# Train Dataset
Features_train = training_dataset[feature_names]  # Features
Features_train=(Features_train-Features_train.mean())/Features_train.std()
Concepts_train = OneHotEncodedDataset[feature_names]
Classes_train = training_dataset["label"]
# Test Dataset
Features_test = testing_dataset[feature_names]
Features_test=(Features_test-Features_test.mean())/Features_test.std()
Concepts_test = OneHotEncodedDataset_test[feature_names]
Classes_test = testing_dataset["label"]
# # Check final dataset sizes
print(f"Features_train shape: {Features_train.shape}, Concepts_train shape: {Concepts_train.shape}, Classes_Train shape: {Classes_train.shape}")
print(f"Features_test shape: {Features_test.shape}, Concepts_test shape: {Concepts_test.shape},  Classes_test shape: {Classes_test.shape}")

#print(f"Concepts_train shape before normalization: {Concepts_train.shape}")print(f"Expected shape: ({training_dataset.shape[0]}, {number_of_important_features})")


xgb_classifier = xgb.XGBClassifier(random_state=seed)
xgb_classifier.fit(Features_train, Classes_train)
classes_predictions = xgb_classifier.predict(Features_test)
baseline_xgb_accuracy = accuracy_score(classes_predictions, Classes_test.values.astype(int))
baseline_xgb_f1_score = f1_score(Classes_test.values.astype(int), classes_predictions, average='weighted')
print("BASELINE Classifier F1 Score:", baseline_xgb_accuracy)
print("BASELINE Classifier Accuracy:", baseline_xgb_f1_score)


feature2concept_classifier = xgb.XGBRegressor(tree_method="hist",multi_strategy="multi_output_tree", random_state=seed)
feature2concept_classifier.fit(Features_train, Concepts_train)
concept_predictions_train = feature2concept_classifier.predict(Features_train)

concept2class_xgb_classifier = xgb.XGBClassifier(random_state=seed)
concept2class_xgb_classifier.fit(concept_predictions_train,Classes_train)

print("Train Acc")
concept_predictions_train = feature2concept_classifier.predict(Features_train)
classes_predictions = concept2class_xgb_classifier.predict(concept_predictions_train)
xgb_accuracy = accuracy_score(classes_predictions, Classes_train.values.astype(int))
xgb_f1_score = f1_score(Classes_train.values.astype(int), classes_predictions, average='weighted')
print("FPC Classifier F1 Score:", xgb_f1_score)
print("FPC Classifier Accuracy:", xgb_accuracy)

print("Test Acc")
concept_predictions = feature2concept_classifier.predict(Features_test)
# xgb_accuracy = concept_accuracy(concept_predictions, Concepts_test.values)
# print("FPC PCAs Accuracy:", xgb_accuracy)
classes_predictions = concept2class_xgb_classifier.predict(concept_predictions)
CBN_xgb_accuracy = accuracy_score(classes_predictions, Classes_test.values.astype(int))
CBN_xgb_f1_score = f1_score(Classes_test.values.astype(int), classes_predictions, average='weighted')
print("FPC Classifier F1 Score:", CBN_xgb_accuracy)
print("FPC Classifier Accuracy:", CBN_xgb_f1_score)

print(f"Accuracy Difference: {baseline_xgb_accuracy - CBN_xgb_accuracy:.4f}")
print(f"F1 Score Difference: {baseline_xgb_f1_score - CBN_xgb_f1_score:.4f}")

np.save(f"{DATASET_PATH}_CBN_seed_{seed}_F{number_of_important_features}.npy", [CBN_xgb_accuracy, CBN_xgb_f1_score])
np.save(f"{DATASET_PATH}_Baseline_seed_{seed}_F{number_of_important_features}.npy", [baseline_xgb_accuracy, baseline_xgb_f1_score])