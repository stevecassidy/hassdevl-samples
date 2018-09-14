"""
Extracting named entities from the Trove Govt. Gazettes
"""

import re
from typing import List, Any

import spacy
from spacy.matcher import Matcher
import pandas as pd
import datetime


def tag_name_address(doc: spacy.tokens.Doc, start: int, end: int) -> list:
    """Given a document span corresponding to a name and address
    identify the entities LAST, FIRST, ADDRESS
    Return a list of tuples (start, end, tag)"""

    state = 'L'
    last = [start, start]
    first = [0, 0]
    addr = [0, 0]
    pos = start
    for tok in doc[start:end]:
        if tok.text == ',':
            if state is 'L':
                state = 'F'
                first = [pos+1, pos+1]
            elif state is 'F':
                state = 'A'
                addr = [pos+1, pos+1]
        else:
            if state is 'L':
                last[1] += 1
            elif state is 'F':
                first[1] += 1
            else:
                addr[1] += 1
        pos += 1

    res = [(first[0], first[1], 'FIRSTNAME'),
           (last[0], last[1], 'LASTNAME'),
           (addr[0], addr[1], 'ADDR'),
           ]

    return res


def tag_dates(nlp, doc):
    """Identify dates in a document by defining some patterns
    Return a list of tags (start, end, 'DATE')
    """
    matcher = Matcher(nlp.vocab)
    matcher.add('DATE', None,
                [{'SHAPE': 'd.d.dd'}],
                [{'SHAPE': 'd,d.dd'}],
                [{'SHAPE': 'd.dd.dd'}],
                [{'SHAPE': 'dd.d.dd'}],
                [{'SHAPE': 'dd.dd.dd'}],
                [{'SHAPE': 'dd.d.dddd'}],
                [{'SHAPE': 'dd.dd.dddd'}],
                [{'SHAPE': 'd.d.dddd'}],
                [{'SHAPE': 'd.dd.dddd'}],
                [{'SHAPE': 'dd'}, {'IS_SPACE': True}, {'SHAPE': 'd.dd'}],
                [{'SHAPE': 'dd'}, {'IS_SPACE': True}, {'SHAPE': 'dd.dd'}],
                [{'IS_DIGIT': True}, {'IS_SPACE': True}, {'IS_DIGIT': True}, {'IS_SPACE': True}, {'IS_DIGIT': True}]
                )
    dates = matcher(doc)
    # generate a list of tuples for each date matched
    dates = [(m[1], m[2], 'DATE') for m in dates]

    return dates


def tag_document(nlp, doc):
    """Given a Spacy document doc corresponding to a Govt. Gazzette "Naturalisation",
     tag the names, addresses and dates
    in the document
    Return a list of tags (start, end, tag)
    """

    dates = tag_dates(nlp, doc)
    current = 0
    tags = []
    for start, end, tag in dates:
        prefix = True
        # skip leading space and punctuation

        for idx in range(current, start):
            if prefix and (doc[idx].is_space or doc[idx].is_punct):
                # skip
                pass
            else:
                current = idx
                break
        # now go tag the name and address in this segment
        tags.extend(tag_name_address(doc, current, start))
        current = end

    tags.extend(dates)
    return tags


def trove_naturalisation_text(article: dict) -> str:
    """Given an article from Trove representing a Govt. Gazette
    article on Naturalisation, return the text of the article
    minus the 'header' part - that is just the body containing
    the list of names and addresses.
    """

    lines: List[str] = re.findall('<span>([^<]+)</span>', article['articleText'])
    text: str = ""
    secretary: bool = False
    for line in lines:
        if not secretary:
            if "Secretary" in line:
                secretary = True
        else:
            text += "\n" + line

    return text


def tag_row(row: pd.Series, nlp: object) -> list:
    """Run the nlp process over a row of a dataframe corresponding
    to an article (must have an element 'text')
    return a list of tags for this article"""

    doc = nlp(row.text)
    tags = tag_document(nlp, doc)
    etags = []
    for start, end, tag in tags:
        etags.append((start, end, tag, str(doc[start:end])))

    return etags


def extract_records(articles: pd.DataFrame) -> pd.DataFrame:
    """Given a dataframe containing tags, extract name/address
    records and return a new dataframe with one record per
    row containig the name and address and supporting
    tags"""

    records = []
    for i, row in articles.iterrows():
        records.extend(tags_to_records(row))
    records = pd.DataFrame(records)
    return records


def tags_to_records(row: pd.Series) -> list:
    """Given a Series (row of dataframe) containing an entry 'tags' that is
    a list of tags (start, end, tag, text)
    in sorted document order, generate a list of name/address records
    return a list of dictionaries
    {'id': article id,
     'support': [(tag tuples)], # tag tuples used to create the record
     'first', 'last', 'address', 'date', 'datestring'   # fields in the record
    }
    """

    # expect FIRSTNAME, LASTNAME, ADDR, DATE
    # dump a new record every time we see a date
    records = []
    record = {'id': row.id, 'support': []}
    for tag in sorted(row.tags):
        text = tag[3].strip().replace('\n', ' ')
        if tag[2] == 'FIRSTNAME':
            record['support'].append(tag)
            record['first'] = text
        elif tag[2] == 'LASTNAME':
            record['support'].append(tag)
            record['last'] = text
        elif tag[2] == 'ADDR':
            record['support'].append(tag)
            record['address'] = text
        elif tag[2] == 'DATE':
            record['support'].append(tag)
            record['datestring'] = text
            record['date'] = parsedate(text)
            if 'first' in record and 'last' in record and 'address' in record:
                records.append(record)
            # otherwise discard
            record = {'id': row.id, 'support': []}
    return records


def parsedate(datestring: str) -> str:
    """Parse a date string into a uniform format 1926-03-12"""

    # replace punctuation with spaces
    datestring = datestring.replace(".", " ").replace(",", " ")
    parts = datestring.split()
    if len(parts) == 3:
        day, month, year = parts
        if len(year) == 2:
            year = "19"+year
        if year.startswith("19") and len(month) <= 2 and len(day) <= 2:
            # dstr = "%s-%s-%s" % (year, month, day)
            try:
                if len(month) == 1:
                    month = "0"+month
                if len(day) == 1:
                    day = "0"+day
                d = datetime.date(int(year), int(month), int(day))
                return d.isoformat()
            except ValueError:
                pass
    return "1900-01-01"


def filter_records(records: list) -> tuple:
    """Given a list of annotation records, remove those that
    have bad data in them
     - numbers in first or last name
     - have a defaulted date field (couldn't parse date)
     - address contains 'formerly' (probably contains more than one record)
     - address is more than 70 chars (probably contains more than one record)
     Return a tuple containing two
     lists (filtered, rejected)"""

    filtered = []
    rejected = []
    for record in records:
        name = record['first'] + record['last']
        if re.search('[0-9]', name) or record['date'] == "1900-01-01" or 'formerly' in record['address'] or len(record['address']) > 70 :
            rejected.append(record)
        else:
            filtered.append(record)

    return filtered, rejected


def valid_record(record: pd.Series) -> bool:
    """Is this a valid record?"""

    name = record['first'] + record['last']
    return not( re.search('[0-9]', name) or \
                record['date'] == "1900-01-01" or \
                'formerly' in record['address'] or \
                len(record['address']) > 70 )

