import fitz
#https://pymupdf.readthedocs.io/en/latest/tutorial.html
import re
from csv import writer

#transcript file names 
pdf_names = ["01_a_ksiazka.pdf", 
             "01_b_ksiazka.pdf", 
             "01_c_ksiazka.pdf", 
             "01_d_ksiazka.pdf", 
             "01_e_ksiazka.pdf", 
             "01_f_ksiazka.pdf", 
             "01_g_ksiazka.pdf", 
             "01_h_ksiazka.pdf", 
             "01_i_ksiazka.pdf", 
             "01_j_ksiazka.pdf", 
             "01_k_ksiazka.pdf", 
             "01_l_ksiazka.pdf", 
             "01_m_ksiazka.pdf",
             "02_a_ksiazka.pdf",
             "02_b_ksiazka.pdf",
             "03_ksiazka.pdf"
             ]

#directory path for transcripts
path = "transcripts/"

#speaking person template: ("\n wyrazy:\n")
pattern_speaker = re.compile(r"[^\n]*[A-ZŻŹĆĄŚĘŁÓŃ][a-zżźćńółęąś]+:\n")

#template for different bugs and defectis finding
pattern_bugs1 = re.compile(r"[^\n]*[A-ZŻŹĆĄŚĘŁÓŃ][a-zżźćńółęąś]+\n")
pattern_bugs2 = re.compile(r"\(([^\)]+)\)")

#individual starting and stoping pages for each file (plus common header margin)
start_page = [27, 4, 4, 4, 6, 6, 4, 6, 4, 6, 6, 6, 6, 6, 6, 4]
stop_page = [0, 0, 1, 1, 3, 3, 3, 2, 2, 4, 2, 3, 4, 2, 5, 4]
header_margin = 80

#dictionary of speeches: 
#Poseł Name Surname: [Speeches List]
sejm_dic = {}

#filling the speeches dictionary
for i, file in enumerate(pdf_names):

    doc_txt = ""
    with fitz.open(path+file) as pdf:
        for page in pdf.pages(start = start_page[i], stop = (pdf.page_count - stop_page[i])):
            clip = page.rect #rectangle coordinates of each page
            clip.y0 += header_margin   #margins definition
            doc_txt += "".join(page.get_text(clip=clip)) #reading page
    
    #removing of common footer (at the end of each file)
    doc_txt = doc_txt.replace("TŁOCZONO Z POLECENIA MARSZAŁKA SEJMU RZECZYPOSPOLITEJ POLSKIEJ PL ISSN 0867-2768. Cena 6,30 zł + 5% VAT", "")
    
    #removing bugs/defects
    doc_txt = pattern_bugs1.sub("", doc_txt)
    
    speakers = pattern_speaker.findall(doc_txt)   #find patterns and save them
    speakers = [item[:-2] for item in speakers]   #removing two last symbols of each speaker's line: ':\n'
    
    #filling the sejm_dic
    for count, speaker in enumerate(speakers):
        
        #removing prefixes placed before name and surname of speaker
        doc_txt = doc_txt.split(speaker+':\n', 1)[1]
        
        if count < len(speakers)-1: #if it is not the last speaker
            quote = doc_txt.split(speakers[count+1]+':\n', 1)[0] #copy all text (up to the next speaker)
        else:
            quote = doc_txt #everything that remains is the text of speach (for the last speaker)

        
        #if speaker is new, add a key to the dictionary and an empty list of speeches
        if (speaker not in sejm_dic):
            sejm_dic[speaker] = []
        
        #remowing dashes and endlines
        quote = quote.replace('-\n', '')
        quote = quote.replace(' \n', ' ')
        quote = quote.replace('\n', ' ')
        
        #removing textes in brackets
        quote = pattern_bugs2.sub("", quote)
        
        #update the speech list
        sejm_dic[speaker].append(quote)


#dictinary standarization (Poseł Name Surname -> Name Surname, excluding other speakers)
sejm_dic_tmp = sejm_dic.copy()
for key in sejm_dic_tmp.keys():
    #Excluding marshalls, vice-marshalls etc
    if ("Marszałek" in key) or ("Wicemarszałek" in key) or ("Sprawozdawca" in key) or ("Sekretarz" in key):
        del sejm_dic[key]
    elif "Poseł " in key: #removing "Poseł" prefixes
        new_key = key.replace("Poseł ", "")
        
        if (new_key not in sejm_dic.keys()):
            sejm_dic[new_key] = []
        
        sejm_dic[new_key].extend(sejm_dic[key])
        del sejm_dic[key]

#invidudual corrections for special cases (untypical surnames)
sejm_dic["Magdalena Małgorzata Kołodziejczak"] = sejm_dic.pop("Kołodziejczak")
sejm_dic["Agnieszka Wojciechowska van Heukelom"] = sejm_dic.pop("van Heukelom")

#removing rest of wrong keys (without first capital letter)
sejm_dic_tmp.clear()
sejm_dic_tmp = sejm_dic.copy()
for key in sejm_dic_tmp:
    if key[0].islower():
        del sejm_dic[key]

#loading party membership from txt file for each speaker
#creation of new, support dictionary: parties_dic
###########
#Parties names: 
#PiS (Prawo i Sprawiedliwosc)
#KO (Koalicja Obywatelska)
#TD (Trzecia Droga)
#NL (Lewica)
#K (Konfederacja)
###########
parties_dic = {}
with open("deputies_parties.txt",encoding = "utf-8") as txt:
    for line in txt:
        #newline removing
        line = line.replace("\n", "")
        #adding "Name Surname": "Party"
        parties_dic[line.split(",")[0]] = line.split(",")[1]

###sejm_dic contains speaches of deputy in a form of: Name Surname : List of Speeches###
###generating csv file plik csv, where rows looks like: Speech; Deputy; Party
#speakers not included in deputies_parties.txt are stored in people_x list - they must be checked manually
people_X = []
with open("sejm.csv", "w", newline='', errors = "ignore") as file:
    csv_writer = writer(file, delimiter = ";")
    csv_writer.writerow(["Speach", "Person", "Party"])
    for key in sejm_dic.keys():
        if key not in parties_dic.keys():
            people_X.append(key)
        else:
            for value in sejm_dic[key]:
                csv_writer.writerow([value, key, parties_dic[key]])
