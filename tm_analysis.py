#################################
#STATISTICS AND DATA EXPLORATION#
#################################

import pandas as pd
import re
import spacy
import matplotlib.pyplot as plt
from collections import Counter

#loading spaCy language model
nlp = spacy.load("pl_core_news_sm")

#loading csv as pandas dataframe
sejm_df = pd.read_csv("sejm.csv", sep = ';', encoding = "windows-1250", encoding_errors = "ignore")

#separate dataframes for each party
PiS_df = sejm_df[sejm_df["Party"]=="PiS"]
KO_df = sejm_df[sejm_df["Party"]=="KO"]
TD_df = sejm_df[sejm_df["Party"]=="TD"]
NL_df = sejm_df[sejm_df["Party"]=="NL"]
K_df = sejm_df[sejm_df["Party"]=="K"]

#number of seats for each party - for scaling purposes
man = [194, 157, 65, 26, 18]

#Number of speeches
print(f"""
      ***Number of speeches in corpus:***
      Prawo i Sprawiedliwość = {PiS_df.shape[0]}
      Koalicja Obywatelska =  = {KO_df.shape[0]}
      Trzecia Droga = {TD_df.shape[0]}
      Lewica = {NL_df.shape[0]}
      Konfederacja  = {K_df.shape[0]}
      """)

print(f"""
      ***Number of speeches in corpus normalized by number of seats:***
      Prawo i Sprawiedliwość = {round(PiS_df.shape[0]/man[0], 2)}
      Koalicja Obywatelska =  = {round(KO_df.shape[0]/man[1], 2)}
      Trzecia Droga = {round(TD_df.shape[0]/man[2], 2)}
      Lewica = {round(NL_df.shape[0]/man[3], 2)}
      Konfederacja  = {round(K_df.shape[0]/man[4], 2)}
      """)
      
      
##LETTERS AND WORDS STATISTICS (CORPUS)##
#Number of letters histograms
plt.figure()
sejm_df["Speach"].str.len().hist()
#plt.savefig('sejm_hist1.png', dpi=300)
plt.figure()
sejm_df["Speach"].str.len().hist(bins = range(0, 5000, 100))
#plt.savefig('sejm_hist2.png', dpi=300)

#letter statistics
print("Number of letters statistics (per speech):")
print(sejm_df["Speach"].str.len().describe())

#Number of words histograms
plt.figure()
sejm_df["Speach"].str.split().map(lambda x: len(x)).hist()
plt.xlabel("Number of words")
plt.ylabel("Number of speeches")
#plt.savefig('sejm_hist3.png', dpi=300)
plt.figure()
sejm_df["Speach"].str.split().map(lambda x: len(x)).hist(bins = range(0, 1000, 50))
plt.xlabel("Number of words")
plt.ylabel("Number of speeches")
#plt.savefig('sejm_hist4.png', dpi=300)

#words statistics
print("Number of words statistics (per speech):")
print(sejm_df["Speach"].str.split().map(lambda x: len(x)).describe())


##LETTERS AND WORDS STATISTICS (BY PARTY)##
#Number of letters histograms
plt.figure(figsize = (15, 10), tight_layout = True)
plt.subplot(321)
PiS_df["Speach"].str.len().hist(bins = range(0, 5000, 100), color = "b")
plt.legend(["PiS"], fontsize = 20)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.subplot(322)
KO_df["Speach"].str.len().hist(bins = range(0, 5000, 100), color = "g")
plt.legend(["KO"], fontsize = 20)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.subplot(323)
TD_df["Speach"].str.len().hist(bins = range(0, 5000, 100), color = "m")
plt.legend(["TD"], fontsize = 20)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.subplot(324)
NL_df["Speach"].str.len().hist(bins = range(0, 5000, 100), color = "r")
plt.legend(["L"], fontsize = 20)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.subplot(325)
K_df["Speach"].str.len().hist(bins = range(0, 5000, 100), color = "k")
plt.legend(["K"], fontsize = 20)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)

#Number of words histograms
plt.figure(figsize = (15, 10), tight_layout = True)
plt.subplot(321)
PiS_df["Speach"].str.split().map(lambda x: len(x)).hist(bins = range(0, 1000, 50), color = "b")
plt.xlabel("Number of words", fontsize = 20)
plt.ylabel("Number of speeches", fontsize = 20)
plt.legend(["PiS"], fontsize = 20)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.subplot(322)
KO_df["Speach"].str.split().map(lambda x: len(x)).hist(bins = range(0, 1000, 50), color = "g")
plt.xlabel("Number of words", fontsize = 20)
plt.ylabel("Number of speeches", fontsize = 20)
plt.legend(["KO"], fontsize = 20)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.subplot(323)
TD_df["Speach"].str.split().map(lambda x: len(x)).hist(bins = range(0, 1000, 50), color = "m")
plt.xlabel("Number of words", fontsize = 20)
plt.ylabel("Number of speeches", fontsize = 20)
plt.legend(["TD"], fontsize = 20)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.subplot(324)
NL_df["Speach"].str.split().map(lambda x: len(x)).hist(bins = range(0, 1000, 50), color = "r")
plt.xlabel("Number of words", fontsize = 20)
plt.ylabel("Number of speeches", fontsize = 20)
plt.legend(["L"], fontsize = 20)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.subplot(325)
K_df["Speach"].str.split().map(lambda x: len(x)).hist(bins = range(0, 1000, 50), color = "k")
plt.xlabel("Number of words", fontsize = 20)
plt.ylabel("Number of speeches", fontsize = 20)
plt.legend(["K"], fontsize = 20)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
#plt.savefig('parties_hist.png', dpi=300)

#Number of words in speech median
plt.figure()
medians = [PiS_df["Speach"].str.split().map(lambda x: len(x)).median(), KO_df["Speach"].str.split().map(lambda x: len(x)).median(),
           TD_df["Speach"].str.split().map(lambda x: len(x)).median(), NL_df["Speach"].str.split().map(lambda x: len(x)).median(),
            K_df["Speach"].str.split().map(lambda x: len(x)).median()]
plt.bar(["PiS", "KO", "TD", "L", "K"], medians, color = ["b", "g", "m", "r", "k"])
plt.xlabel("Party", fontsize = 15)
plt.ylabel("Number of words median", fontsize = 15)
#plt.savefig('parties_med.png', dpi=300)




##USED WORDS STATISTICS (CORPUS)##
#how many of most frequently used words we want to analyze
first_n = 40

#lowercasing and removing puncations - sejm_df_cleaned creation
pattern_punctuation = re.compile(r"[\.,!?:\–…]")
speach_cleaned = []
for row in sejm_df["Speach"]:
    row_lower = "".join([char.lower() for char in row])
    speach_cleaned.append(pattern_punctuation.sub("", row_lower))
    
sejm_df_cleaned = pd.DataFrame({"Speach": speach_cleaned,
                                   "Person": sejm_df["Person"].copy(),
                                   "Party": sejm_df["Party"].copy()})

#main corpus
corpus_sejm = []
for x in sejm_df_cleaned["Speach"].str.split():
    corpus_sejm.extend(x)

#plotting most frequently used words - still with stop words
count_sejm = Counter(corpus_sejm).most_common()

x, y= [], []
for word,count in count_sejm[:first_n]:
        x.append(word)
        y.append(count)
        
plt.figure(figsize = (10, 25))
plt.barh([i for i in range(first_n, 0, -1)], y, tick_label = x)
plt.yticks(fontsize = 25)
plt.xticks(fontsize = 20)
plt.xlabel("Number", fontsize = 25)
plt.ylabel("Word", fontsize = 25)
#plt.savefig('sejm_words.png', dpi=300)

#main corpus - without stop words
stop_w = spacy.lang.pl.stop_words.STOP_WORDS

#adding words to stop words list ("Panie Marszałku, Wysoka Izbo")
stop_w.add('panie')
stop_w.add('marszałku')
stop_w.add('wysoka')
stop_w.add('izbo')

corpus_sejm_cleaned = [word for word in corpus_sejm if word not in stop_w]

#plotting most frequently used words - without stop words
count_sejm_cleaned = Counter(corpus_sejm_cleaned).most_common()
x, y= [], []

for word,count in count_sejm_cleaned[:first_n]:
        x.append(word)
        y.append(count)
        
plt.figure(figsize = (10, 25))
plt.barh([i for i in range(first_n, 0, -1)], y, tick_label = x)
plt.yticks(fontsize = 25)
plt.xticks(fontsize = 20)
plt.xlabel("Number of speeches", fontsize = 25)
plt.ylabel("Word", fontsize = 25)
#plt.savefig('sejm_words_stop.png', dpi=300)

##USED WORDS STATISTICS (BY PARTY)##

#support variables for plotting
names = ["PiS", "KO", "TD", "L", "K"]
colors = ["b", "g", "m", "r", "k"]
first_n = 40

#corpuses for parties
corpus_PiS = []
corpus_KO = []
corpus_TD = []
corpus_L = []
corpus_K = []

for index, row in sejm_df_cleaned.iterrows():
    if row[2] == "PiS":
        corpus_PiS.extend(row[0].split())
    elif row[2] == "KO":
        corpus_KO.extend(row[0].split())
    elif row[2] == "TD":
        corpus_TD.extend(row[0].split())
    elif row[2] == "NL":
        corpus_L.extend(row[0].split())
    elif row[2] == "K":
        corpus_K.extend(row[0].split())

#stop words removing
corpus_PiS = [word for word in corpus_PiS if word not in stop_w]
corpus_KO = [word for word in corpus_KO if word not in stop_w]
corpus_TD = [word for word in corpus_TD if word not in stop_w]
corpus_L = [word for word in corpus_L if word not in stop_w]
corpus_K = [word for word in corpus_K if word not in stop_w]

#plotting most frequently used word by party
count_PiS = Counter(corpus_PiS).most_common()
count_KO = Counter(corpus_KO).most_common()
count_TD = Counter(corpus_TD).most_common()
count_L = Counter(corpus_L).most_common()
count_K = Counter(corpus_K).most_common()

counts = [count_PiS, count_KO, count_TD, count_L, count_K]

plt.figure(figsize = (50, 25), tight_layout = True)
for i, c in enumerate(counts):
    x, y= [], []
    for word,count in c[:first_n]:
            x.append(word)
            y.append(count)
            
    plt.subplot(1, 5, i+1)
    plt.barh([i for i in range(first_n, 0, -1)], y, tick_label = x, color = colors[i])
    plt.yticks(fontsize = 40)
    plt.xticks(fontsize = 20)
    plt.legend([names[i]], fontsize = 50)

#finding frequent words for each party
universal_words = []
for word, count in count_PiS[:first_n]:
    if (word in [a for (a, b) in count_KO[:first_n]]):
        if (word in [a for (a, b) in count_TD[:first_n]]):
            if (word in [a for (a, b) in count_L[:first_n]]):
                if (word in [a for (a, b) in count_K[:first_n]]):
                    universal_words.append(word)
print(f"Words present in top{first_n} for all parties (there are {len(universal_words)} of them):")
for w in universal_words:
    print(w)

print("\n")

#finding unique words for each party
unique_PiS = []
unique_KO = []
unique_TD = []
unique_L = []
unique_K = []
uniques = [unique_PiS, unique_KO, unique_TD, unique_L, unique_K]

for i, c in enumerate(counts):
    #creating of counts_tmp list wihout current element
    counts_tmp = counts.copy()
    counts_tmp.pop(i)
    
    #loop checking the current word in other lists
    for word, count in c[:first_n]:
        unique = True
        
        for c2 in counts_tmp:
            if word in [a for (a, b) in c2[:first_n]]:
                unique = False
                break
        
        if unique:
            uniques[i].append(word)

for i, u in enumerate(uniques):
    print(f"Unique words in top{first_n} for party {names[i]}:")
    for u2 in u:
        print(u2)
    print("\n")