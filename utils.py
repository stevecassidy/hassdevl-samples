"""
Utilities for preprocessing data prior to NER 

Author: Steve Cassidy <steve.cassidy@mq.edu.au>
"""

import pandas as pd
import xml.etree.ElementTree as ET
import geocoder
import json


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
    texts = dv['occurrenceRemarks']
    
    return pd.DataFrame({'text': texts})


def read_ftp_xml(filename):
    """Read texts from the TEI XML format
    export from From The Page. Return a dataframe
    containing the text in the 'text' column"""
    
    tree = ET.parse(filename)
    root = tree.getroot()
    texts = []
    for para in root.iter('{http://www.tei-c.org/ns/1.0}p'):
        if para.text:
            texts.append(para.text)
    return pd.DataFrame({'text': texts})


def geolocate_locations(loc, countryBias=['AU']):
    """Given a data frame with a column 'placename' run
    a geolocation service over each placename. 
    Add new columns to the data frame for 
    'address', 'country', 'lat', 'long'
    returns the new dataframe
    """

    GEONAMES_KEY = secret('geonames')
    geo = []
    for place in loc['placename']:
        g = geocoder.geonames(place, key=GEONAMES_KEY, countryBias=countryBias)
        if g:
            result = {'lat': g.lat, 'lng': g.lng, 'address': g.address, 'country': g.country}
        else:
            result = {'lat': 0, 'lng': 0, 'address': '', 'country': ''}
        geo.append(result)

    geo = pd.DataFrame(geo)
    return pd.concat([loc, geo], axis=1)