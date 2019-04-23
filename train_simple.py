import pandas
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from dsppkeras.datasets import dspp
from utils import lettercode2onehot
from keras.preprocessing.sequence import pad_sequences
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import scikitplot as skplt

'''
    This program uses several classification algorithms to predict the secondary structure of amino acid sequences
    in a dataset (the dataset is located in datasets/2018-06-06-ss.cleaned.csv. The classification algorithms
    determine whether an amino acid is part of a helix, sheet, or loop (a reduced DSPP format). The algorithms
    are given windows (or set lengths of amino acid sequences) in which the center amino acid corresponds with
    an output structure. The sliding-window scheme is used as a way of providing context to the output of a
    certain secondary structure. For example:

    The amino acid Leucine in the following 15 size window GRRASVE|L|PLGPLPP corresponds to the structure of a
    loop. The surrounding amino acids around the L amino acid give context to the algorithm. In this study,
    a maximum sequence length of 30 and a window size of 15 are used.
'''

def sliding_window(sequence,winSize,step=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""
    
    # Verify the inputs
    try: it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")
    
    # Pre-compute number of chunks to emit
    numOfChunks = int(((len(sequence)-winSize)/step)+1)
    
    # Do the work
    for i in range(0,numOfChunks*step,step):
        yield sequence[i:i+winSize]

def load_training_data(input_seqs, target_seqs, sliding_window_size):
    '''
    load_training_data: load binary classification inputs and outputs for secondary protein structure
    parameters: input_seqs(numpy.ndarray of string input sequences),
                target_seqs(numpy.ndarray of string target sequences),
                sliding_window_size(number of amino acids in sliding window scheme)
    returns: an x input vector and y target vector
    '''
    
    assert len(input_seqs) == len(target_seqs)

    x = []
    y = []
    
    center_pos = int(sliding_window_size / 2)
    
    for seq, sst3 in zip(input_seqs, target_seqs):
        seq_chunks = sliding_window(seq, sliding_window_size)

        target_index = center_pos
        
        for chunk in seq_chunks:
            target = sst3[target_index]

            output = 0
            
            if target == 'H':
                output = 0
            elif target == 'E':
                output = 1
            elif target == 'C':
                output = 2
            else:
                raise Exception('Unknown sst3 structure found!')

            x.append(lettercode2onehot(chunk))
            y.append(output)
            
            target_index += 1
        
    return x, y

# The maximum amount of amino acids in a sequence to be analyzed
maxlen_seq = 30
# The size of the sliding window
sliding_window_size = 15

# Read data frame
df = pandas.read_csv('datasets/2018-06-06-ss.cleaned.csv')

# Get sequences that match our parameters
input_seqs, target_seqs = df[['seq', 'sst3']][(df.len <= maxlen_seq) & (df.len >= sliding_window_size) & (~df.has_nonstd_aa)].values.T

# Print the number of sequences to be trained and tested upon
print('Number of sequences analyzed: %d' % (len(input_seqs)))

# Load the training/testing set
x, y = load_training_data(input_seqs, target_seqs, sliding_window_size)

# Random partitioning seed
seed = 8

# List of models to be evaluated
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(probability=True)))
models.append(('MLP', MLPClassifier()))

# List of results of each model
results = []

# List of model names
names = []

# Metric used for comparison
scoring = 'accuracy'

# List of class names used for precision-recall curve plot
class_labels = [ 'Helix', 'Sheet', 'Loop' ]

viridis = cm.get_cmap('viridis', 12)

for name, model in models:
    # First do a KFold comparison
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    # Print statistics
    print(msg)

    # Next do a normal training fit and plot precision-recall curve
    model.fit(x, y)
    probas = model.predict_proba(x)
    skplt.metrics.plot_precision_recall(y, probas, title=(name + ' Precision-Recall Curve'), class_labels=class_labels, cmap=viridis)
    plt.savefig('figures/' + name + 'RecallPrecisionPlot' + '.png', bbox_inches='tight')
	
# Boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Secondary Structure Algorithm Classification Accuracy Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results, showfliers=False)
ax.set_xticklabels(names)
plt.xlabel('Algorithm')
plt.ylabel('Classification Accuracy')
plt.savefig('figures/MLAlgorithmsSSClassificationComparison.png', bbox_inches='tight')

