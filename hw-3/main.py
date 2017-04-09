import nltk
from string import punctuation
import wikipedia

text = None
with open('text.txt', 'r') as f:
    text = f.read()

text = text.decode('utf-8').encode('ascii', 'ignore')

sentences = nltk.sent_tokenize(text)
tokens = [nltk.word_tokenize(sent) for sent in sentences]
tagged = [nltk.pos_tag(sent) for sent in tokens]

print(tagged)

tokens = nltk.word_tokenize(text)
tagged = nltk.pos_tag(tokens)

entities = []

def extractEntities(ne_chunked):
    data = {}
    for entity in ne_chunked:
        if isinstance(entity, nltk.tree.Tree):
            text = " ".join([word for word, tag in entity.leaves()])
            ent = entity.label()
            data[text] = ent
            if (text not in entities):
                entities.append(text)
        else:
            continue
    return data

ne_chunked = nltk.ne_chunk(tagged, binary=False)

print(extractEntities(ne_chunked))

entity = []
for tagged_entry in tagged:
    if((not entity and tagged_entry[1].startswith("JJ")) or (entity and tagged_entry[1].startswith("NN"))):
        entity.append(tagged_entry)
    else:
        entity = []
    if(len(entity)== 2):
        if (entity[1][0] not in entities):
            entities.append(entity[1][0])
        print(entity[1][0])


for entity in entities:
    try:
        page = wikipedia.page(entity)
    except:
        continue

    first_sentence = nltk.sent_tokenize(page.summary)[0]
    tokens = nltk.word_tokenize(first_sentence)
    tagged = nltk.pos_tag(tokens)


    phrase = []
    for tagged_entry in tagged:
        if(len(phrase)== 2 and tagged_entry[1].startswith("NN")):
            phrase.append(tagged_entry)
            print(entity + ' is a ' + " ".join(p[0] for p in phrase))
            break
        if((not phrase and tagged_entry[1].startswith("JJ")) or (phrase and tagged_entry[1].startswith("NN"))):
            phrase.append(tagged_entry)
        else:
            phrase = []


