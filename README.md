**TEXT ANALYZIS AND CLASSIFICATION FOR SPEECHES IN POLISH PARLIAMENT**

This project was created during my postgraduate, data science-related studies on Warsaw University of Technology. It was created to  develop the text corpus of deputies speeches in Polish Parliament (10. term), perform analysis of them and then build 3 independent NLP models for speeches classification. Based on input text, each model tries to predict the deputy party membership of a speaker. 
All code is written in Python.

Content list: 
- deputies_parties.txt – text file containing a party membership for each deputy mentioned in corpus (sejm.csv). Template for each row: 
Name Surname,Party
- transcripts/ - directory containing official transcripts downloaded from Polish Government Website (https://sejm.gov.pl/Sejm10.nsf/stenogramy.xsp). Files are used for corpus creation
- pdf_import.py – python script that reads all pdf files stored in transcripts, and then creates the sejm.csv file
- README.md - brief project description (this file)
- sejm.csv – csv file containing 3 columns: speech, speaker and party membership. Each speech read from transcript forms a separate row
- tm_analysis.py – python script performing a statistic analysis of given corpus (stored in sejm.csv): words count, most frequently used words etc.
- tm_classification.py – python script that builds 3 different text classificators (all from sklearn python library): SGDClassifier, MultinominalNB and Logistic Regression. The role of each one of them is to assign one of 5 classes (PiS, KO, TD, L and K) to given speech. Used classes corresponds with parties in Polish Parliament (PiS – Prawo i Sprawiedliwość, KO – Koalicja Obywatelska, TD – Trzecia Droga, L – Lewica, K – Konfederacja)
- tm_pis_detector.py – python script similar to tm_classification.py, but this time there are only two classes to assign: PiS and non-PiS (other party than PiS)

Used Python libraries:
- collections
- csv
- fitz
- matplotlib
- numpy
- pandas
- re
- sklearn
- spacy

Transcripts are are periodically updated (last update: 21.01.2024). 
