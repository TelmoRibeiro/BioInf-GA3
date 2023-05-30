import sys
import pandas as pd
import numpy as np
from itertools import product 

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score



def read_Fasta (filename):
    """
    Read the sequences from fasta files. 
    Return the data as a dictionary where the id of the sequence is the key and the sequence is the value.
    """

    from re import sub, search

    dic={}
    sequence = None
    id = None
    
    fh = open(filename)

    for line in fh:
        if search(">.*", line):
                if sequence is not None and id is not None and sequence != "":
                    dic[id]=sequence
                id = search("(?<=\|)[^|]+", line).group(0)  # Extract ID between '|' characters             
                sequence = ""
        else:
            if sequence is None: return None
            else: sequence += sub("\s","",line)

    if sequence is not None and id is not None and sequence != "":
                    dic[id]=sequence

    fh.close()

    return dic


def word_to_kmer(k):

    result = []
    alphabet  = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

    kmers = list(product(alphabet,repeat=k))
    for kmer in kmers:
        kmer = "".join(kmer)
        result.append(kmer)

    return result


def generate_kmers(seq, k):
    """
    Given a sequence returns a dictionary that contains the frequency of each k-mer in the sequence.
    Normalize the frequency by the total number of 2-mers in the sequence Fi,j = Ci,j /|2-mers|
    """

    kmer_freq={}
    kmer_n=len(seq)-k+1

    for i in range(kmer_n):
        kmer=seq[i:i+k]

        if kmer in kmer_freq:
            kmer_freq[kmer]+=1
        else:
            kmer_freq[kmer]=1

    for k in kmer_freq:
        kmer_freq[k]=kmer_freq[k]/kmer_n

    return kmer_freq

def generate_FFP_df(file_name_1,file_name_2, k):
    """
    Creates and fills a pandas dataframe, with the FFP values for all the sequences in both input files. 
    The columns should have the 400 Fi,j dinucleotide values while the rows should correspond to 
    each sequences in the two datasets identified by the respective sequence id.
    """

    fam1= read_Fasta(file_name_1)
    fam2= read_Fasta(file_name_2)

    mers= word_to_kmer(k)

    df = pd.DataFrame(columns=mers+ ["Class"])

    for id in fam1:
        FFP=generate_kmers(fam1[id],k)
        FFP["Class"]=0
        df.loc[id] = pd.Series(FFP)


    for id in fam2:
        FFP=generate_kmers(fam2[id],k)
        FFP["Class"]=1
        df.loc[id] = pd.Series(FFP)

    df = df.fillna(0)
    df['Class'] = df['Class'].astype(int)
    
    return df


def classification(df):
    """
    Apply 3 Machine Learning algorithms to predict if the type of the protein 
    - methods: Random Forests, SVM, NaiveBayes
    - metrics:  accuracy, recall, precision, F1-score
    - Stratified 10-fold cross-validation 
    Return a dataframe with the average and standard deviation across the 10 folds of all the metrics of all the methods
    """
     
    # Split the dataframe in features and target (protein class)
    X = df.iloc[:, :-1] 
    y = df['Class']     

    # Create a cross-validation object (Stratified 10-fold)
    scv = StratifiedKFold(n_splits=10, shuffle=True, random_state=23)

    # Define the metrics
    scoring = ['accuracy', 'recall', 'precision', 'f1']

    # Define a dataframe for storing the scores
    scores_df= pd.DataFrame()

    # Random Forest 
    rforest_model = RandomForestClassifier() 
    rforest_results = cross_validate(rforest_model, X, y, cv=scv, scoring=scoring, return_train_score=False)

    rforest_scores={
        "mean_accuracy" : np.mean(rforest_results['test_accuracy']),
        "mean_recall" : np.mean(rforest_results['test_recall']),
        "mean_precision" : np.mean(rforest_results['test_precision']),
        "mean_f1" : np.mean(rforest_results['test_f1']),
        "std_accuracy" : np.std(rforest_results['test_accuracy']),
        "std_recall" : np.std(rforest_results['test_recall']),
        "std_precision" :np.std(rforest_results['test_precision']),
        "std_f1" : np.std(rforest_results['test_f1'])        
        }

    new_row = pd.DataFrame(rforest_scores, index=["Random Forest"])
    scores_df = pd.concat([scores_df, new_row])

    # SVM
    svm_model = SVC()
    svm_results = cross_validate(svm_model, X, y, cv=scv, scoring=scoring, return_train_score=False)

    svm_scores = {
        "mean_accuracy" : np.mean(svm_results['test_accuracy']),
        "mean_recall" : np.mean(svm_results['test_recall']),
        "mean_precision" : np.mean(svm_results['test_precision']),
        "mean_f1" : np.mean(svm_results['test_f1']),
        "std_accuracy" : np.std(svm_results['test_accuracy']),
        "std_recall" : np.std(svm_results['test_recall']),
        "std_precision" :np.std(svm_results['test_precision']),
        "std_f1" : np.std(svm_results['test_f1'])        
    }

    new_row = pd.DataFrame(svm_scores, index=["SVM"])
    scores_df = pd.concat([scores_df, new_row])

    # Naive Bayes
    nb_model = GaussianNB()
    nb_results = cross_validate(nb_model, X, y, cv=scv, scoring=scoring, return_train_score=False)

    nb_scores = {
        "mean_accuracy" : np.mean(nb_results['test_accuracy']),
        "mean_recall" : np.mean(nb_results['test_recall']),
        "mean_precision" : np.mean(nb_results['test_precision']),
        "mean_f1" : np.mean(nb_results['test_f1']),
        "std_accuracy" : np.std(nb_results['test_accuracy']),
        "std_recall" : np.std(nb_results['test_recall']),
        "std_precision" : np.std(nb_results['test_precision']),
        "std_f1" : np.std(nb_results['test_f1'])        
    }

    new_row = pd.DataFrame(nb_scores, index=["Naive Bayes"])
    scores_df = pd.concat([scores_df, new_row])
   
    return(scores_df)



def main():

    np.random.seed(42)

    for i in range(1,len(sys.argv),2):
        if sys.argv[i]=="-a":
            file_name_1 = sys.argv[i+1]
        elif sys.argv[i]=="-b":
            file_name_2 = sys.argv[i+1]
        elif sys.argv[i]=="-k":
            k= int(sys.argv[i+1])
    
             
    dataframe=generate_FFP_df(file_name_1,file_name_2, k)
    print("FFP VALUES")
    print(dataframe)
    with open('FFP_VALUES.txt', 'w') as f:
        print('FFP VALUES:\n', dataframe.to_string(), file=f)
    
    classification_scores=classification(dataframe)
    print("CLASSIFICATION")
    print(classification_scores)


main()