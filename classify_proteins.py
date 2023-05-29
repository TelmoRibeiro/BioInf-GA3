# retrives both files and k from the console arguments # 
def parseArguments(arguments):
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

# extracts the rough data from a fasta file # 
def read_fasta(filename):
    file = open(filename, 'r')
    data = file.read()
    file.close()
    return data

# extracts the mapping (id:sequence) from a fasta file #
def extract_map(filename):
    from re import findall
    data = read_fasta(filename)
    identifiers = findall(">sp\|[^\|]*\|", data)
    sequences   = findall("\n[^>]*>",  data+">")                # data + ">" allows to capture the last sequence
    if len(identifiers) != len(sequences):
        raise Exception("#identifiers != #sequences")
    identifiers_map = {}
    for index in range(len(identifiers)):
        identifier = identifiers[index][4:-1]                   # strips ">sp||" from the identifier
        sequence   = sequences[index][1:-2].replace("\n","")    # strips "\n\n>" from the sequence and erases the '\n' between lines
        identifiers_map[identifier] = sequence
    return identifiers_map

# gets all kmer's permutations # 
def get_all_kmers(k=2):
    from itertools import product 
    kmers    = []
    alphabet = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    permutations = list(product(alphabet,repeat=k))
    for kmer in permutations:
        kmer = "".join(kmer)    # casts the kmer to a clean string | string() does not have the intended outcome here
        kmers.append(kmer)
    return kmers

# gets the normalized frequency of each kmer in the sequence # 
def get_kmers_frequency(sequence, k=2):
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

# used in generate_FFP_dataframes(...) and classification #
import pandas

# creates and fills a pandas dataframe with the FFP values for all the sequences #
#                     columns = 400 Fi,j dinucleotide values                     #
#  rows = sequences in the two datasets identified by the respective identifier  # 
def generate_FFP_dataframe(fileA, fileB, k=2):
    familyA  = extract_map(fileA)
    familyB  = extract_map(fileB)
    kmers = get_all_kmers(k)
    dataframe = pandas.DataFrame(columns = kmers + ["Class"])
    for identifier in familyA:
        FFP = get_kmers_frequency(familyA[identifier], k)
        FFP["Class"] = 1
        dataframe.loc[identifier] = pandas.Series(FFP)
    for identifier in familyB:
        FFP = get_kmers_frequency(familyB[identifier], k)
        FFP["Class"] = 0
        dataframe.loc[identifier] = pandas.Series(FFP)
    dataframe = dataframe.fillna(0)
    dataframe['Class'] = dataframe['Class'].astype(int)
    return dataframe

# Applies Random Forests, SVM and NaiveBayes in order to predict the type of protein #
#                   metrics: accuracy, recall, precision, F1-score                   #
#                        Stratified 10-fold cross-validation                         # 
def classification(dataframe):
    import numpy
    from   sklearn.ensemble         import RandomForestClassifier
    from   sklearn.svm              import SVC
    from   sklearn.naive_bayes      import GaussianNB
    from   sklearn.model_selection  import StratifiedKFold, cross_validate
    from   sklearn.metrics          import accuracy_score, recall_score, precision_score, f1_score
    # Split the dataframe in features and target (protein class) #
    x   = dataframe.iloc[:,:-1]
    y   = dataframe['Class']     
    scv = StratifiedKFold(n_splits=10, shuffle=True, random_state=23)   # cross-validation object (Stratified 10-fold)
    scoring = ['accuracy', 'recall', 'precision', 'f1']                 # metrics
    scores_dataframe= pandas.DataFrame()                                # dataframe for storing the scores
    # Random Forest #
    rforest_model   = RandomForestClassifier() 
    rforest_results = cross_validate(rforest_model, x, y, cv=scv, scoring=scoring, return_train_score=False)
    rforest_scores  = {
        "mean_accuracy"     : numpy.mean(rforest_results['test_accuracy']),
        "mean_recall"       : numpy.mean(rforest_results['test_recall']),
        "mean_precision"    : numpy.mean(rforest_results['test_precision']),
        "mean_f1"           : numpy.mean(rforest_results['test_f1']),
        "std_accuracy"      : numpy.std (rforest_results['test_accuracy']),
        "std_recall"        : numpy.std (rforest_results['test_recall']),
        "std_precision"     : numpy.std (rforest_results['test_precision']),
        "std_f1"            : numpy.std (rforest_results['test_f1'])        
    }
    new_row = pandas.DataFrame(rforest_scores, index=["Random Forest"])
    scores_dataframe = pandas.concat([scores_dataframe, new_row])
    # SVM #
    svm_model   = SVC()
    svm_results = cross_validate(svm_model, x, y, cv=scv, scoring=scoring, return_train_score=False)
    svm_scores  = {
        "mean_accuracy"     : numpy.mean(svm_results['test_accuracy']),
        "mean_recall"       : numpy.mean(svm_results['test_recall']),
        "mean_precision"    : numpy.mean(svm_results['test_precision']),
        "mean_f1"           : numpy.mean(svm_results['test_f1']),
        "std_accuracy"      : numpy.std (svm_results['test_accuracy']),
        "std_recall"        : numpy.std (svm_results['test_recall']),
        "std_precision"     : numpy.std (svm_results['test_precision']),
        "std_f1"            : numpy.std (svm_results['test_f1'])        
    }
    new_row = pandas.DataFrame(svm_scores, index=["SVM"])
    scores_dataframe = pandas.concat([scores_dataframe, new_row])
    # Naive Bayes #
    nb_model   = GaussianNB()
    nb_results = cross_validate(nb_model, x, y, cv=scv, scoring=scoring, return_train_score=False)
    nb_scores = {
        "mean_accuracy"     : numpy.mean(nb_results['test_accuracy']),
        "mean_recall"       : numpy.mean(nb_results['test_recall']),
        "mean_precision"    : numpy.mean(nb_results['test_precision']),
        "mean_f1"           : numpy.mean(nb_results['test_f1']),
        "std_accuracy"      : numpy.std (nb_results['test_accuracy']),
        "std_recall"        : numpy.std (nb_results['test_recall']),
        "std_precision"     : numpy.std (nb_results['test_precision']),
        "std_f1"            : numpy.std (nb_results['test_f1'])        
    }
    new_row = pandas.DataFrame(nb_scores, index=["Naive Bayes"])
    scores_dataframe = pandas.concat([scores_dataframe, new_row])
    return scores_dataframe

# program's flow controller "
def main(arguments):
    (fileA,fileB,k) = parseArguments(arguments)
    dataframe       = generate_FFP_dataframe(fileA,fileB,k)
    print(f"FFP Values:\n{dataframe}")
    output_file = open('FFP_Values.txt', 'w')
    print(f"FFP Values:\n{dataframe.to_string()}", file=output_file)
    output_file.close()
    classification_scores = classification(dataframe)
    print(f"Classification:\n{classification_scores}")
    return

from sys import argv
if __name__ == "__main__": main(argv)