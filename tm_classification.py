#######################
#CLASSIFICATION MODELS#
#######################

import pandas as pd
import spacy
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


##DATA PREPARATIONS##

#maximum number of words in single speech (median: 92, quartile 3/4: 138) - words are counted from the beginning
#when variable words_max in commented, complete speeches are taken into account
#words_max = 50

#minimum number of words (below this threshold speech is omitted)
#words_min = 10

#parties names
names = ["PiS", "KO", "TD", "NL", "K"]

#loading csv file (containing complete corpus)
data_df = pd.read_csv("sejm.csv", usecols = ["Speach", "Party"], sep = ';', encoding = "windows-1250", encoding_errors = "ignore")

#loading spaCy language model and adding additional stop words
nlp = spacy.load("pl_core_news_sm")

stop_w = spacy.lang.pl.stop_words.STOP_WORDS
stop_w.add("panie")
stop_w.add("marszałku")
stop_w.add("wysoka")
stop_w.add("izbo")

#data cleaning and transforming (lower case letters, removing punctuations, stop words etc)
pattern_punctuation = re.compile(r"[\.,!?:\–…]")
x_cleaned = []
for row in data_df["Speach"]:
    row_lower = "".join([char.lower() for char in row])
    row_npun = pattern_punctuation.sub("", row_lower)
    if "words_max" in locals():
        row_nstop = [word for word in row_npun.split(" ") if word not in stop_w]
        row_limited = " ".join(word for word in row_nstop[:words_max])
    else:
        row_limited = " ".join([word for word in row_npun.split(" ") if word not in stop_w])
    x_cleaned.append(row_limited)

#cleaned text as new column
data_df.insert(2, "Cleaned", x_cleaned)

#if words_min is defined, indexes of rows to be removed are saved in index_to_remove list
index_to_remove = []
if "words_min" in locals():
    for i, row in enumerate(data_df["Cleaned"]):
        if len(row.split()) < words_min:
            index_to_remove.append(i)

#specyfing the input and output data (x and y)
x_df = pd.DataFrame({"Cleaned": data_df["Cleaned"].drop(index = index_to_remove)})
y_df = pd.DataFrame({"Party": data_df["Party"].drop(index = index_to_remove)})


#lemmatization
x_lem = []
for doc in nlp.pipe(x_df["Cleaned"]):
    x_lem.append(" ".join([token.lemma_ for token in doc ]))

x_df.insert(1, "Lemmatized", x_lem)

#defying of training and test datasets
x_train, x_test, y_train, y_test = train_test_split(x_df["Lemmatized"], y_df, test_size=0.3, random_state = 44)

#distribution of target variable
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))

#vectorization - TfidfVectorizer
tfid = TfidfVectorizer(encoding = "windows-1250")
x_train_tfid = tfid.fit_transform(x_train)
x_test_tfid = tfid.transform(x_test)

#vectorization - CountVectorizer
count_vec = CountVectorizer(lowercase = False, encoding = "windows-1250", ngram_range = (2, 3))
x_train_count = count_vec.fit_transform(x_train)
x_test_count = count_vec.transform(x_test)


##SGDClassifier##

#SVM model training
SGDC_SVM = SGDClassifier(loss = "hinge", penalty = "l2", alpha=0.0001, tol = 0.001, random_state = 44)
SGDC_SVM.fit(x_train_tfid, np.ravel(y_train))
print(f"Number of iterations during model training: {SGDC_SVM.n_iter_}")

#prediction for train dataset
y_pred_train_svm = SGDC_SVM.predict(x_train_tfid)

#prediction for test dataset
y_pred_test_svm = SGDC_SVM.predict(x_test_tfid)

#prediction quality
SGDC_SVM_train_score = accuracy_score(y_train, y_pred_train_svm)
SGDC_SVM_test_score = accuracy_score(y_test, y_pred_test_svm)
print(f"Accuracy score for SGDC_SVM for train dataset: {round(SGDC_SVM_train_score, 3)}")
print(f"Accuracy score for SGDC_SVM for test dataset: {round(SGDC_SVM_test_score, 3)}")

cls_rep_SGDC_SVM = classification_report(y_test, y_pred_test_svm)
print(cls_rep_SGDC_SVM)

conf_mat_svm = confusion_matrix(y_test, y_pred_test_svm, labels = names)
ConfusionMatrixDisplay(conf_mat_svm, display_labels = names).plot()


##MultinomialNB##

#model learning
MNB = MultinomialNB(alpha = 1, force_alpha = "warn", fit_prior = True, class_prior = None)
MNB.fit(x_train_count, np.ravel(y_train))

#prediction for train dataset
y_pred_train_mnb = MNB.predict(x_train_count)

#prediction for test dataset
y_pred_test_mnb = MNB.predict(x_test_count)

#prediction quality
MNB_train_score = accuracy_score(y_train, y_pred_train_mnb)
MNB_test_score = accuracy_score(y_test, y_pred_test_mnb)
print(f"Accuracy score for MultinomialNB for train dataset: {round(MNB_train_score, 3)}")
print(f"Accuracy score for MultinomialNB for test dataset: {round(MNB_test_score, 3)}")

cls_rep_MNB = classification_report(y_test, y_pred_test_mnb)
print(cls_rep_MNB)

conf_mat_mnb = confusion_matrix(y_test, y_pred_test_mnb, labels = names)
ConfusionMatrixDisplay(conf_mat_mnb, display_labels = names).plot()



##LogisticRegression##

#model learning
LR = LogisticRegression(random_state = 44, C = 1e3, max_iter=1000, class_weight = "balanced")
LR.fit(x_train_tfid, np.ravel(y_train))
print(f"Number of iterations during model training: {LR.n_iter_}")

#prediction for training dataset
y_pred_train_lr = LR.predict(x_train_tfid)

#prediction for test dataset
y_pred_test_lr = LR.predict(x_test_tfid)

#prediction quality
LR_train_score = accuracy_score(y_train, y_pred_train_lr)
LR_test_score = accuracy_score(y_test, y_pred_test_lr)
print(f"Accuracy score for Logistic Regression for train dataset: {round(LR_train_score, 3)}")
print(f"Accuracy score for Logistic Regression for test dataset: {round(LR_test_score, 3)}")

cls_rep_LR = classification_report(y_test, y_pred_test_lr)
print(cls_rep_LR)

conf_mat_lr = confusion_matrix(y_test, y_pred_test_lr, labels = names)
ConfusionMatrixDisplay(conf_mat_lr, display_labels = names).plot()