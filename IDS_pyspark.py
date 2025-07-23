import pandas as pd
import re
import numpy as np
import kagglehub,string
from pathlib import Path
from pyspark import SparkFiles
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import pyshark


spark = SparkSession.builder.appName("example").getOrCreate()

def reduceSampleCount(data,new_perc_size,column_name,column_value=-1):
    new_data=None
    if column_value!=-1:
        new_data = data[data[column_name] == column_value]
    else:
        new_data = data[data[column_name]]
    new_samples_count = int(len(new_data) * (new_perc_size/100))
    new_data_sample = new_data.sample(n=new_samples_count, random_state=42)
    full_sampled_data=None
    if column_value!=-1:
        full_sampled_data = pd.concat([new_data_sample, data[data[column_name] != column_value]])
    else:
        full_sampled_data = pd.concat([new_data_sample, data[data[column_name]]])
    full_sampled_data.reset_index(drop=True, inplace=True)
    return full_sampled_data
def captureData(dataFilter="",timeout=15):
    capture = pyshark.LiveCapture(
        interface='ethmon0',
        display_filter=dataFilter) #replace with your interface(use listInterfaces() to view all available)
    #capture.set_display_filter('http')
    if (timeout!=0):
        capture.sniff(timeout=timeout)
    packets = [pkt for pkt in capture._packets]
    capture.close()
    return packets

def hypertune():
    paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [int(x) for x in np.linspace(start = 10, stop = 50, num = 3)]) \
    .addGrid(rf.maxDepth, [int(x) for x in np.linspace(start = 5, stop = 25, num = 3)]) \
    .build()

    return paramGrid

def print_confusion_matrix(y_test,y_pred):
    from sklearn.metrics import confusion_matrix 
    import matplotlib.pyplot as plt
    import numpy as np
            
    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
            
    # Compute percentages
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
    # Plotting the confusion matrix with percentages
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_perc, cmap='Blues')
            
    # Add percentage values to the heatmap cells
    thresh = cm_perc.max() / 2.0
    for a in range(cm_perc.shape[0]):
            for b in range(cm_perc.shape[1]):
                plt.text(j, i, format(cm_perc[i, j], '.3f'), ha='center', va='center',
                    color='white' if cm_perc[i, j] > thresh else 'black')
            
    plt.title('Confusion Matrix %')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar()
    plt.show()
def print_imp_features(cv_model):
    best_rf_model = cv_model.bestModel.stages[-1]
    importances = best_rf_model.featureImportances
    print("Feature Importances:")
    for feature, importance in zip(feature_list, importances):
        print(f"{feature}: {importance:.4f}")

hyperparam_tune=False
if __name__=="__main__":
    print("start")
    path = kagglehub.dataset_download("kk0105/cicids2017")


    home_dir = Path.home()
    f = open(str(home_dir)+"\.cache\kagglehub\datasets\kk0105\cicids2017\\versions\\2\Week_filtered.csv",
             encoding='utf-8')
    data=pd.read_csv(str(home_dir)+"\.cache\kagglehub\datasets\kk0105\cicids2017\\versions\\2\Week_filtered.csv")
    #data=data.rename(columns={"Label":"label"})
    
    for name in data.columns:
        if bool(re.search(r'[^a-zA-Z0-9]', name)):
            #print(name)
            data=data.rename(columns={name:name.translate(str.maketrans('', '', string.punctuation))})
##    print(features)
##    index = features.index('label')
##    features.remove(index)
##    data.iloc[:, features].replace([np.inf], np.nan,inplace=True)
##    #data[:, data.columns != 'label']= np.nan
##    na_rows = np.where(np.isna(data).any(axis=1))[0]
##    data=np.delete(data,na_rows)
    data=data.head(300) #reduce data size for faster processing
    #viewDataStats(data)
    spark = SparkSession.builder.appName('example').getOrCreate()
    data=spark.createDataFrame(data)
    features_names=list(data.columns)
    features_names.remove(('Label'))
    sIndexer=StringIndexer(inputCol="Label",outputCol="label")
    assembler = VectorAssembler(inputCols=features_names, outputCol="features")
    rf = RandomForestClassifier(labelCol="label", featuresCol="features")
    pipeline = Pipeline(stages=[sIndexer,assembler, rf])

    #Stages replaced by pipeline
##    data=sIndexer.fit(data).transform(data)
##    data = assembler.transform(data)
    print(data.columns)
    train_data, test_data = data.randomSplit([0.7, 0.3], seed=42)
    p_model=None
    test_data.show(10)
    if hyperparam_tune:
        paramGrid=hypertune()
        cross_validator = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy"),
                              numFolds=5, seed=42)
        p_model = cross_validator.fit(train_data)
    else:
        p_model = pipeline.fit(train_data)
    
    preds = p_model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")

    accuracy = evaluator.evaluate(preds)
    print("Test set accuracy ="+str(accuracy))
    c_model = p_model.stages[-1]
    importances = c_model.featureImportances
    print("Feature Importances:", c_model)
