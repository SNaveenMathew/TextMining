# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:38:06 2017

@author: naveen.nathan
"""

from nltk.parse.stanford import StanfordDependencyParser
from json import load
#from nltk.parse.corenlp import CoreNLPParser

# Initializating global variables
parser_cfg = load(open("parser.cfg"))
parser_jar = parser_cfg["parser.jar"]
parser_models_jar = parser_cfg["models.jar"]
dependency_parser = StanfordDependencyParser(path_to_jar = parser_jar,
                                                 path_to_models_jar = parser_models_jar)
#parser = CoreNLPParser()

# Available options:
# 1) Stanford parser
# 2) CoreNLP parser (not working - need to check if it is same as Stanford parser)

# Purpose: Runs dependency parser on string
# Input: String
# Output: Parse tree
def run_stanford_parser(text):
    result = dependency_parser.raw_parse(text)
    dep = result.__next__()
    return dep

# Currently not working
#def run_core_nlp_parser(text):
#    return parser.parse(text)
