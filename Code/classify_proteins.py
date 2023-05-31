from sys import argv
import pandas as pd
import numpy as np
from itertools import product 

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def parseArguments(arguments):
    """
    Retrieves both files and k from the console arguments.
    """
    for index in range(1, len(arguments), 2):
        if   arguments[index] == "-a": # first  file
            fileA = arguments[index+1]
        elif arguments[index] == "-b": # second file
            fileB = arguments[index+1]
        elif arguments[index] == "-k": # k-mers
            k = int(arguments[index+1])
        else:
            raise Exception(f"{arguments[index]} is not a valid argument") 
    return (fileA, fileB, k)

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


def get_all_kmers(k=2):
    """
    Gets all kmer's permutations.
    """
    kmers    = []
    alphabet = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    permutations = list(product(alphabet,repeat=k))
    for kmer in permutations:
        kmer = "".join(kmer)    # casts the kmer to a clean string | string() does not have the intended outcome here
        kmers.append(kmer)
    return kmers

def get_kmers_frequency(sequence, k=2):
    """
    Given a sequence returns a dictionary that contains the frequency of each k-mer in the sequence.
    Normalize the frequency by the total number of 2-mers in the sequence Fi,j = Ci,j /|2-mers|.
    """
    kmers_frequency = {}
    number_kmers = len(sequence) - k + 1        # number of possible kmers in the sequence
    for index in range(number_kmers):
        kmer = sequence[index:index+k]
        if kmer in kmers_frequency:
            kmers_frequency[kmer] += 1
        else:
            kmers_frequency[kmer]  = 1
    for kmer in kmers_frequency:
        kmers_frequency[kmer] /= number_kmers   # normalizes the frequency by the number of kmers
    return kmers_frequency


def generate_FFP_dataframe(fileA, fileB, k=2):
    """
    #  Creates and fills a pandas dataframe with the FFP values for all the sequences. #
    #                     columns = 400 Fi,j dinucleotide values                       #
    #  rows = sequences in the two datasets identified by the respective identifier    #
    """
    familyA  = read_Fasta(fileA)
    familyB  = read_Fasta(fileB)
    kmers = get_all_kmers(k)
    dataframe = pd.DataFrame(columns = kmers + ["Class"])
    for identifier in familyA:
        FFP = get_kmers_frequency(familyA[identifier], k)
        FFP["Class"] = 1
        dataframe.loc[identifier] = pd.Series(FFP)
    for identifier in familyB:
        FFP = get_kmers_frequency(familyB[identifier], k)
        FFP["Class"] = 0
        dataframe.loc[identifier] = pd.Series(FFP)
    dataframe = dataframe.fillna(0)
    dataframe['Class'] = dataframe['Class'].astype(int)
    return dataframe


def classification(dataframe):
    """
    # Applies Random Forests, SVM and NaiveBayes in order to predict the type of protein. #
    #                   metrics: accuracy, recall, precision, F1-score.                   #
    #                        Stratified 10-fold cross-validation.                         # 
    """
    # Split the dataframe in features and target (protein class) #
    x   = dataframe.iloc[:,:-1]
    y   = dataframe['Class']     
    scv = StratifiedKFold(n_splits=10, shuffle=True, random_state=23)   # cross-validation object (Stratified 10-fold)
    scoring = ['accuracy', 'recall', 'precision', 'f1']                 # metrics
    scores_dataframe= pd.DataFrame()                                    # dataframe for storing the scores
    
    # Random Forest #
    rforest_model   = RandomForestClassifier() 
    rforest_results = cross_validate(rforest_model, x, y, cv=scv, scoring=scoring, return_train_score=False)
    rforest_scores  = {
        "mean_accuracy"     : np.mean(rforest_results['test_accuracy']),
        "mean_recall"       : np.mean(rforest_results['test_recall']),
        "mean_precision"    : np.mean(rforest_results['test_precision']),
        "mean_f1"           : np.mean(rforest_results['test_f1']),
        "std_accuracy"      : np.std (rforest_results['test_accuracy']),
        "std_recall"        : np.std (rforest_results['test_recall']),
        "std_precision"     : np.std (rforest_results['test_precision']),
        "std_f1"            : np.std (rforest_results['test_f1'])        
    }
    new_row = pd.DataFrame(rforest_scores, index=["Random Forest"])
    scores_dataframe = pd.concat([scores_dataframe, new_row])

    # SVM #
    svm_model   = SVC()
    svm_results = cross_validate(svm_model, x, y, cv=scv, scoring=scoring, return_train_score=False)
    svm_scores  = {
        "mean_accuracy"     : np.mean(svm_results['test_accuracy']),
        "mean_recall"       : np.mean(svm_results['test_recall']),
        "mean_precision"    : np.mean(svm_results['test_precision']),
        "mean_f1"           : np.mean(svm_results['test_f1']),
        "std_accuracy"      : np.std (svm_results['test_accuracy']),
        "std_recall"        : np.std (svm_results['test_recall']),
        "std_precision"     : np.std (svm_results['test_precision']),
        "std_f1"            : np.std (svm_results['test_f1'])        
    }
    new_row = pd.DataFrame(svm_scores, index=["SVM"])
    scores_dataframe = pd.concat([scores_dataframe, new_row])

    # Naive Bayes #
    nb_model   = GaussianNB()
    nb_results = cross_validate(nb_model, x, y, cv=scv, scoring=scoring, return_train_score=False)
    nb_scores = {
        "mean_accuracy"     : np.mean(nb_results['test_accuracy']),
        "mean_recall"       : np.mean(nb_results['test_recall']),
        "mean_precision"    : np.mean(nb_results['test_precision']),
        "mean_f1"           : np.mean(nb_results['test_f1']),
        "std_accuracy"      : np.std (nb_results['test_accuracy']),
        "std_recall"        : np.std (nb_results['test_recall']),
        "std_precision"     : np.std (nb_results['test_precision']),
        "std_f1"            : np.std (nb_results['test_f1'])        
    }
    new_row = pd.DataFrame(nb_scores, index=["Naive Bayes"])
    scores_dataframe = pd.concat([scores_dataframe, new_row])
    return scores_dataframe


def main(arguments):

    np.random.seed(42)

    (fileA,fileB,k) = parseArguments(arguments)
    
    dataframe=generate_FFP_dataframe(fileA,fileB, k)
    print(f"FFP Values:\n{dataframe}")
    output_file = open('FFP_Values.txt', 'w')
    print(f"FFP Values:\n{dataframe.to_string()}", file=output_file)
    output_file.close()

    #class_counts = dataframe['Class'].value_counts()
    #print(class_counts[0])
    #print(class_counts[1])

    classification_scores = classification(dataframe)
    print(f"Classification:\n{classification_scores}")


if __name__ == "__main__": main(argv)