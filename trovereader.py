"""
Code for reading data from Trove archive files in XML format

Trove archives are XML files, potentially very large with one <record> element
per record or document. We provide a class that can be used to iterate
over the records in a file as Python dictionaries.

"""
from typing import Dict, List

from lxml import etree


def trove_parser(xmlfile: str) -> List:
    """Read records corresponding to documents from an XML
    file exported from Trove, yield records one at
    a time to the caller.
    Each record is a dictionary with metadata properties.
    All dictionary values are lists.
    """
    context = etree.iterparse(xmlfile, events=('end',), tag='record')

    for event, elem in context:
        record = {}
        for child in elem:
            if child.text:
                if child.tag == 'identifier' and 'linktype' in child.attrib:
                    propname = child.attrib['linktype']
                else:
                    propname = child.tag

                if propname in record:
                    record[propname].append(child.text.strip())
                else:
                    record[propname] = [child.text.strip()]
        yield record

if __name__=='__main__':

    #xmlfile = "data/nla-advocate-sample.xml"
    xmlfile = "data/nla.obj-573721295_Aborginies_Advocate.xml"
    i = 0
    for record in trove_parser(xmlfile):
        print(record['identifier'][0], len(record['description'][0]))
        i += 1
        if i > 3:
            break
