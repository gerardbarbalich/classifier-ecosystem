import os
import glob
import itertools
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


# define constants
SEL_DATE = '2018-12-14'
DATA_DIR = os.path.join(os.environ['HOME'], f"dns-flag-day-{SEL_DATE}-content")
SPLIT_TEST_SIZE = 0.5
SPLIT_SEED = 2

# import data from WebScan
d = pd.concat([pd.read_pickle(f) for f in glob.glob(os.path.join(DATA_DIR, '*.p'))])

# import list of confirmed fake webshops
df_fake_domains = pd.read_csv('df_fake_domains.csv')
ls_fake_domains = df_fake_domains['domains'].values.tolist()

# group the text by domain
df = d[['domain', 'BodyText']].\
    groupby('domain')['BodyText'].\
    apply(lambda x: " ".\
    join(x))

# label domains
df['label'] = df['domain'].apply(lambda x: 'Yes' if x in ls_fake_domains else 'No')

# define the tockenizer
custom_stop = stopwords.words('english') + \
['nz', 'st', 'www', 'co', 'new', 'zealand', 'nzd', 'us', 'ml', 'javascript']

custom_stop_set = set(custom_stop)

tokenizer = RegexpTokenizer(r'[a-zA-Z]{2,}')

def gen_tokens(text):
    return [w.lower() for w in tokenizer.tokenize(text) if w not in custom_stop_set]

# define classifier components
clf_steps = [('vec_tfidf', TfidfVectorizer(stop_words=custom_stop,
                                           tokenizer=gen_tokens,
                                           max_features=1000,
                                           ngram_range=(1, 2))),
            ('clf', RandomForestClassifier(random_state=42,
                                           max_features='auto',
                                           n_estimators=200))]

# construct the pipeline
clf = Pipeline(clf_steps)

# create train an test datasets
x_train, x_test, y_train, y_test = train_test_split(df['BodyText'],
                                                    df['label'],
                                                    test_size=SPLIT_TEST_SIZE,
                                                    random_state=SPLIT_SEED)

# train the pipeline
clf.fit(x_train, y_train)

# make predictions to test the pipeline
y_pred = clf.predict(x_test).tolist()

# create a confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)

# split the confusion matrix into sperate integers
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()


# print key experimental results
print(cnf_matrix)
print("Training df length: ", len(y_train))

print('Test Recall - % of labelled positives that are correctly predicted')
print("%.3f" % (tp / (tp + fn)))

print('Test Precision - % of predicted positives are actual positives')
print("%.3f" % (tp / (tp + fp)))

print('Test Accuracy - % of predictions that are actually correct')
print("%.3f" % ((tp + tn) / (tn + fn + tp + fp)))

print('F1 Score:')
print("%.3f" % (f1_score(y_test, y_pred, average="macro")))

# check the k-fold cross validation scores
scores = cross_val_score(clf, x_train, y_train, cv=10, scoring='accuracy')
print(scores)


# export the trained pipeline
joblib.dump(clf, 'fake-webshop-pipeline-[date].pkl')
