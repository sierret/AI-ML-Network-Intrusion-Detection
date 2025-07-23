import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
import kagglehub
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pyshark

def viewDataStats(data):
    if (isinstance(data, pd.DataFrame)):
        total_nan = data.isna().sum().sum()
        print("Data Dimensions : %d rows, %d columns" % (data.shape[0], data.shape[1]))
        print("Total Non-Numeric Values : %d " % (total_nan))
        print("Name", "Type", "#Distinct", "NAN Values")
        columns = data.columns
        types = data.dtypes
        unique = data.nunique()
        nan_values = data.isna().sum()
        for i in range(len(data.columns)):
            print(columns[i], types[i], unique[i], nan_values[i])

    else:
        print("Not a Dataframe :" + str(type(data)))
        exit(0)

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

def encodeNanTargetValues(df):
    lEcde = LabelEncoder()
    df.Label = lEcde.fit_transform(df.Label)
    return df

def final_data_clean(chunk,target_name):
    # Replace NaN with average value of feature
    nan_rows = chunk[chunk.isna().any(axis=1)].shape[0]
    chunk.iloc[:, chunk.columns != target_name] = chunk.groupby(target_name).transform(lambda x: x.fillna(x.mean()))
    # Temporary replace inf with NaN for ease of transformation
    chunk = chunk.replace([np.inf], np.nan)
    #Replace inf with max value of feature
    chunk.iloc[:, chunk.columns != target_name] = chunk.groupby(target_name).transform(lambda x: x.fillna(x.max()))

    #Replace neg value with with min pos value of feature
    # Temporary replace negative value with NaN for ease of transformation
    chunk[chunk < 0] = np.nan
    # Replace neg value with pos min value of feature
    chunk.iloc[:, chunk.columns != target_name] = chunk.groupby(target_name).transform(lambda x: x.fillna(x.min()))

##    df=chunk
##    infinity_count = df.isin([np.inf, -np.inf]).sum().sum()
##    # Checking for NaN values
##    null_count = df.isnull().sum().sum()
##    nan_count = df.isna().sum().sum()
##    print(f"Infinity count: {infinity_count}")
##    print(f"Null count: {null_count}")
##    print(f"NaN count: {nan_count}")
    return chunk

def clean_data_by_chunk(data,target_name,chunksize: int=100000,max_val_size=-1):
    new_data = pd.DataFrame()
    for i in range(0, len(data), chunksize):
        chunk = data.iloc[i:i+chunksize, :]
        chunk.reset_index(drop=True,inplace=True)
        cleaned_chunk = final_data_clean(chunk,target_name)
        if (max_val_size!=-1):
            cleaned_chunk = cleaned_chunk[(chunk.astype(float) <= max_val_size).all(axis=1)]
        new_data = pd.concat([new_data, chunk])

    #print(np.isinf(new_data.values).all())
    return new_data
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
                plt.text(b, a, format(cm_perc[a, b], '.3f'), ha='center', va='center',
                         color='white' if cm_perc[a, b] > thresh else 'black')

        plt.title('Confusion Matrix %')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.colorbar()
        plt.show()

if __name__=="__main__":
    print("start")
    path = kagglehub.dataset_download("kk0105/cicids2017")


    home_dir = Path.home()
    f = open(str(home_dir)+"\.cache\kagglehub\datasets\kk0105\cicids2017\\versions\\2\Week_filtered.csv",
             encoding='utf-8')
    data=pd.read_csv(str(home_dir)+"\.cache\kagglehub\datasets\kk0105\cicids2017\\versions\\2\Week_filtered.csv")

    data=data.head(300) #reduce data size for faster processing
    #viewDataStats(data)
    #data=reduceSampleCount(data,"Label",20)

    data=encodeNanTargetValues(data)
    import struct

    f32_size = np.dtype(np.float32).itemsize
    data=clean_data_by_chunk(data,"Label",f32_size)

    X = data.iloc[:, : -1].values #separate first n-1 column data which are features
    #showDataStats(X)
    y = data.iloc[:,  -1].values  #separate last n column data is which is a target column

    
    
    
    # Remove rows with inf values
    
    # Get Idx of rows with inf values
    rows_with_inf = np.where(np.isinf(X).any(axis=1))[0]

    X=np.delete(X,rows_with_inf,axis=0)
    y=np.delete(y,rows_with_inf,axis=0)

    #X=X[np.all(np.isfinite(X), axis=1)]
    #or X=X[np.isfinite(X).all(1)]
    ##    print(np.isinf(X).any())
    ##    inf_values = X_test[np.isinf(X_test)]
    ##    print(inf_values)

    #Replace inf with 0
    #array[~np.isfinite(array)]=0

    X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3, random_state=99)
    clf=None
    #clf = MultinomialNB() #Bayes

    clf = RandomForestClassifier(random_state=42) #Random Forest
    clf.fit(X_train, y_train)


    
    print("Accuracy:"+(str)(clf.score(X_test, y_test)))

    print_confusion_matrix(y_test,clf.predict(X_test))

    from sklearn.model_selection import cross_val_score, KFold  
    kf = KFold(n_splits = 10)  
    score = cross_val_score(clf, X_train, y_train, cv=kf)  
      
    print("K-fold Cross Validation Scores are: ", score)
      
    print("Mean Cross Validation score is: ", score.mean())
