from collections import OrderedDict
import os


class Document(object):

    def __init__(self, fname):
        self.name = fname.split('/')[-1]
        self.url = 'None'
        self.headline = ['headline']
        self.lead = []
        self.sentences = OrderedDict()
        self.tags = dict()
        self.sent_to_speech = dict()
        self.sent_to_event = dict()


def process_doc(textFormatted, TAGS=True):
    #prints sentence-by-sentence
    #print(textFormatted)

    doc = Document("Title")

    sent_to_event = dict()

    f = textFormatted

    for index, line in enumerate(f.splitlines()):

        temp = line.strip().split()

        if line == "":
            continue
        if line[0] == 'H':
            doc.headline = temp[1:]
        if line[0] != 'S':
            continue

        if TAGS:
            sent_to_event[temp[0]] = temp[0]
        # temp, label = temp.split('\t')
        doc.sentences[temp[0]] = temp[1:]

        # Process annotation file
    doc.sent_to_event = sent_to_event
    return doc
