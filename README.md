# AI-ML-Network-Intrusion-Detection
Uses ML techniques to train a model that can determine if/when there is *an attempt* at unauthorized access to a network or device as well as the type of intrusion/unauthorized access

Currently uses Random Forest Classifier and the cicicds2017 dataset to train the model. The model is intended to be used with recorded actvitity from the tshark package to determine if the user's network has been accessed by an unauthorized actor. Function "captureData" has been included as a example on how to capture the data. These model would then be run on the packet captures. Currently, the implementation on how the data is cleaned is left up to the user(eg.as there are many columns that may be removed). Thus, the further step of model predicting real user data is not [yet] implemented.

Note: There are two versions: one dependent on scikit-learn and the other on pyspark  pakage. Either can be used separately.

Prints Confusion Matrix
