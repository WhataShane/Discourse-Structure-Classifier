import random
import torch
from process_file import process_doc
import numpy as np
import os
import torch.optim as optim
import time
from discourse_profiler import Classifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import cluster

import pandas as pd
import numpy as np
import nltk
#run to download sentence tokenizer:
#nltk.download('punkt')

np.random.seed(0)
CUDA = torch.cuda.is_available()

def get_batch(doc, ref_type='headline'):
    ref, sent, lr, ls, out, sids = [], [], [], [], [], []
    # print(sids)
    for sid in doc.sentences:
        if ref_type == 'headline':
            ref.append(doc.headline)
            lr.append(len(doc.headline))
        elif ref_type == 'lead':
            ref.append(doc.lead)
            lr.append(len(doc.lead))
        elif ref_type == 'both':
            ref.append(doc.headline+doc.lead)
            lr.append(len(doc.headline+doc.lead))
        sent.append(doc.sentences[sid])
        ls.append(len(doc.sentences[sid]))
        sids.append(doc.sent_to_event[sid])
    lr = torch.LongTensor(lr)
    ls = torch.LongTensor(ls)
    out = torch.LongTensor(out)
    return ref, sent, lr, ls, out, sids


#takes dataframe, returns all unique article ids as a list
def return_ids(dataframe):
    return dataframe['article'].drop_duplicates().tolist()


#takes dataframe, tokenizes text from "text" sentence-by-sentence, create new col "sent_tokenized"
#WARNING -- breaks if any rows have no text (are empty)
def df_sentence_tokenizer(dataframe):
    dataframe['sent_tokenized'] = dataframe.apply(lambda row: nltk.sent_tokenize(row['text']), axis=1)


#takes an article id and returns the article's text properly formatted as S1, S2, S3...
#must run df_sentence_tokenizer first as format_text depends upon the sent_tokenized col
def format_text(articleId):

    articleText = df.loc[df['article'].isin([str(id)]), 'sent_tokenized'].tolist()
    formattedText = ""

    sentenceCounter = 1

    for paragraph in articleText:
        for sentence in paragraph:
            formattedText += "S"+str(sentenceCounter)+" "+sentence+"\n\n"
            sentenceCounter += 1

    return formattedText


#takes processed doc as input (see process_file.py)
#outputs list of classifications on a sentence-by-sentence basis
#each list item is assigned a number 0 - 8, apply mapperFunction below for text labels
def dump_output(doc):
    discourse_model.eval()
    # for doc in data:
    ref, sent, lr, ls, out, sids = get_batch(doc)
    if CUDA:
        lr = lr.cuda()
        ls = ls.cuda()
    _output, _, _, _ = discourse_model.forward(ref, sent, lr, ls)
    scoresContainer = _output.tolist()

    classifications = []

    for scores in scoresContainer:
        classifications.append( scores.index(max(scores)) )

    return classifications

#maps list with classifications labels
def mapperFunction(classifications):
    dict = ['NA', 'M1', 'M2', 'C1', 'C2', 'D1', 'D2', 'D3', 'D4']
    return [dict[x] for x in classifications]

#converts mapped list to "[C1 - C2 - D1]" format
#truncates list by specifications
#howMany is inclusive
def finalFormat(mappedClassifications, startIndex, howMany):
    finale = mappedClassifications[startIndex:(startIndex+howMany)]
    return " - ".join(finale)


def df_classifier(dataframe, id, classifications):

    startingPoint = 0

    for index, row in dataframe.iterrows():
        #print("==new entry==")

        if (row['article'] == id):
            dataframe.loc[index,'classifs'] = finalFormat(classifications, startingPoint, len(row['sent_tokenized']))
            #print( "start: "+str(startingPoint), "# of: "+str(len(row['sent_tokenized'])) )
            startingPoint += len(row['sent_tokenized'])


if __name__ == '__main__':
    SPEECH = 0
    if SPEECH:
        out_map = {'NA':0, 'Speech':1}
    else:
        out_map = {'NA':0,'Main':1,'Main_Consequence':2, 'Cause_Specific':3, 'Cause_General':4, 'Distant_Historical':5,
        'Distant_Anecdotal':6, 'Distant_Evaluation':7, 'Distant_Expectations_Consequences':8}

    torch.manual_seed(0)
    if CUDA:
        torch.cuda.manual_seed(0)
    random.seed(0)
    discourse_model = Classifier({'num_layers': 1, 'hidden_dim': 512, 'bidirectional': True, 'embedding_dim': 1024,
                        'dropout': 0.5, 'out_dim': len(out_map)})
    if CUDA:
        discourse_model = discourse_model.cuda()

    discourse_model.load_state_dict(torch.load('discourse_profiling.pt', map_location=torch.device('cpu')))

    test_data = []

    #import ================================
    filename = "./tester.csv"
    #to chunk data, use param nrows=50
    df = pd.read_csv(filename)
    #=======================================

    df_sentence_tokenizer(df)

    ids = return_ids(df)

    for id in ids:
        text = format_text(id) #turns entire article into S1, S2, S3, etc...
        doc = process_doc(text, True) #converts to format model can process
        classsifList = dump_output(doc) #returns each sentence's classifications (as a #) in a list
        classsifList = mapperFunction(classsifList) #maps each number to respective text-classification

        df_classifier(df, id, classsifList)

        df.to_csv('output.csv')

        #print(df)
        #print(id)
        #print(classsifList)
        #print("\n\n")
