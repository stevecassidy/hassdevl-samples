"""
Utilities for preprocessing data prior to NER 

Author: Steve Cassidy <steve.cassidy@mq.edu.au>
"""

import pandas as pd
import xml.etree.ElementTree as ET
import geocoder
import json
import requests
import os
import spacy

# get the spacy models we need, download if it isn't here
model = 'en_core_web_md'
try:
   spacy.load(model)
except IOError:
    spacy.cli.download(model)

nlp = spacy.load(model)

def secret(key):
    """Get the value of 'key' from the file
    'secret.json' in the current directory"""
    
    with open("secret.json") as fd:
        keys = json.load(fd)
        
    if key in keys:
        return keys[key]
    else:
        return "unknown"


def read_digivol_csv(filename):
    """Read a CSV file exported from Digivol
    return a DataFrame with text in the 'text' column"""
    
    dv = pd.read_csv('data/Project-1536729-DwC.csv')
    t = [x.replace('\r\n', '\n') for x in dv['occurrenceRemarks']]
    dv['text'] = t
    dv.drop('occurrenceRemarks', axis=1, inplace=True)
    return dv


def read_ftp_xml(filename):
    """Read texts from the TEI XML format
    export from From The Page. Return a dataframe
    containing the text in the 'text' column and the 'xml:id' attribute for 
    each text (these are pages in the original document)"""
    
    tree = ET.parse(filename)
    root = tree.getroot()
    texts = []
    for para in root.iter('{http://www.tei-c.org/ns/1.0}p'):
        if '{http://www.w3.org/XML/1998/namespace}id' in para.attrib:
            ident = para.attrib['{http://www.w3.org/XML/1998/namespace}id']
        else:
            ident = "unknown"
        # get all text inside the paragraph, ignoring tags
        # replace \r\n sequences with just \n since it confuses spaCy
        text = "".join(para.itertext()).replace('\r\n','\n')
        if text:
            texts.append({'id': ident, 'text': text})
    return pd.DataFrame(texts)


def apply_ner(df, textcol='text', ident='id', keep_entities=[], context=2):
    """Apply SpaCy Named Entity Recognition to texts in a data
    frame.  'textcol' is the column containing text, 'ident' is
    the column containing a document identifier. 
    'context' is the number of words each side of the entity to 
    return as the context of each hit. Resulting
    data frame will have columns """
    
    entities = []

    for i, t in df.iterrows():
        text = t[textcol]
        doc = nlp(text)
        for ent in doc.ents:
            if ent.text.strip() != '' and (not keep_entities or ent.label_ in keep_entities):
                # get the context words, take care at the start and end of the document
                ctx = doc[max(0,ent.start-context):min(len(doc), ent.end+context)]
                ctx = " ".join([w.text for w in ctx])
                d = {'entity': ent.text, 'type': ent.label_, 'context': ctx, 'id': t[ident]}
                entities.append(d)
                
    return pd.DataFrame(entities)


def geolocate_locations(loc, countryBias=['AU']):
    """Given a data frame with a column 'placename' run
    a geolocation service over each placename. 
    Add new columns to the data frame for 
    'address', 'country', 'lat', 'long'
    returns the new dataframe
    """

    GEONAMES_KEY = secret('geonames')
    geo = []
    for place in loc['entity']:
        g = geocoder.geonames(place, key=GEONAMES_KEY, countryBias=countryBias)
        if g:
            result = {'lat': g.lat, 'lng': g.lng, 'address': g.address, 'country': g.country}
        else:
            result = {'lat': 0, 'lng': 0, 'address': '', 'country': ''}
        geo.append(result)

    geo = pd.DataFrame(geo)
    return pd.concat([loc, geo], axis=1)


def trove_query_cached(q: str, n: int=100, cachefile: str='articles.json', force: bool=False) -> pd.DataFrame:
    """Perform a query over trove but use cachefile as a
    cache - if it is already present, read data from it
    instead unless force is True

    Return a Pandas dataframe containing the query results"""

    if not force and os.path.exists(cachefile):

        with open(cachefile) as fd:
            articles = json.load(fd)
    else:
        articles = utils.trove_query(q, n)

    articles = pd.DataFrame(articles)
    articles.index = articles.id

    return articles


def trove_query(q:str, n:int=100) -> dict:
    """A simple Trove API interface, 
    q is a query term, we search the 
    newspaper zone and return
    the decoded JSON response (a Python dictionary)"""
    
    TROVE_API_KEY = secret('trove')
    TROVE_API_URL = "http://api.trove.nla.gov.au/result"
    qterms = {
        'zone': 'newspaper',
        'encoding': 'json',
        'include': 'articleText',
        's': 0,
        'n': n,
        'key': TROVE_API_KEY,
        'q': q
    }
    r = requests.get(TROVE_API_URL, params=qterms).json()
    articles = r['response']['zone'][0]['records']['article']
    remaining = n-100
    while remaining > 0:
        qterms['n'] = remaining
        qterms['s'] += 100
        r = requests.get(TROVE_API_URL, params=qterms)
        r = r.json()
        art = r['response']['zone'][0]['records']['article']
        if len(art) > 0:
            articles.extend(art)
            remaining -= 100
        else:
            # no more articles
            remaining = 0
        
    return articles