# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:33:25 2017

@author: huijing.deng
"""


from math import isnan
#from en_core_web_md import load
from os import environ
from pandas import DataFrame, concat
import langdetect
from autocorrect.nlp_parser import NLP_WORDS
from nltk.corpus import stopwords
from string import punctuation
from re import findall, sub, compile
import pandas as pd
from dateparser import parse


# Other utilities:
# 1) Read text file
# 2) Flatten a double-list into list
# 3) Clean beginning of sentences
# 4) Pick language with highest probability from set of languages
# 5) Check whether language is English with prabability > p (default = 0.5)
# 6) Spell correct (in progress)

patterns = [".*about to release.*", ".*affare top.*", ".*alleged buyback.*",
".*alleged cap* increase.*", ".*alleged spinoff.*", ".*banking secrecy.*",
".*be careful with * info.*", ".*before the announcement.*", ".*bid letters submission planning.*",
".*buy-out in programma.*", ".*buyout planeado? previsto.*", ".*cap* increase plans.*"]
patterns = compile("|".join(patterns))

def read_file(file, in_type = "csv", message_col = "Message"):
    if in_type.lower() == "csv":
        from pandas import read_csv
        return read_csv(file, encoding = "latin1")
    elif in_type.lower() == "excel":
        from pandas import read_excel
        return read_excel(file, encoding = "latin1")
    elif in_type.lower() == "html":
        from pandas import read_html
        try:
            df = read_html(file)
        except:
            df = []
        if type(df) == list and len(df) > 0:
            
            if len(df) == 6:
                length = len(df)
                conversation = []
                language = "null"
                num_of_conversation_turns = 0
            elif len(df) == 7:
                length = len(df)-1
                conversation = get_conversation(df)
                languages = conversation[message_col].apply(detect_language)
                first_language = languages.apply(pick_first_language)
                english_only = first_language.apply(is_english_wp_p)
                total_english = english_only.sum()
                language = "en"
                if total_english <= 2:
                    language = first_language.apply(lambda x: x.lang).value_counts()
                    language = language.index[0]
                num_of_conversation_turns = conversation.shape[0]
                
            
            meta_data = df[0:length]
            meta_data[1] = meta_data[1].T
            meta_data[2] = meta_data[2].T
            import numpy as np
            if meta_data[1][0][0] == "No reviewing has been done":
                d = np.array([["Date","null"],["Action Status","null"],["Reviewer","null"]])
                meta_data[1] = pd.DataFrame(data=d,columns=[0, 1])
            if meta_data[2][0][0] == "No comments have been left":
                d = np.array([["Date","null"],["Comment","null"],["Reviewer","null"]])
                meta_data[2] = pd.DataFrame(data=d,columns=[0, 1])
                
            meta_data = meta_data[:-3] + meta_data[-2:]
            meta_data = concat(meta_data, axis = 0, ignore_index = True)
            timestamp = meta_data[2]
            timestamp = timestamp[timestamp.apply(is_not_nan)].tolist()[0]
            timestamp = str(timestamp).lower()
            meta_data1 = meta_data[1]
            meta_data1.index = meta_data[0]
            
            messageType = str(meta_data1['Message Type:']).lower()
            messageDirection = str(meta_data1['Message Direction:']).lower()
            case = str(meta_data1['Case:']).lower()
            captureDate = str(meta_data1['Capture Date:']).lower()
            itemId = str(meta_data1['Item ID:']).lower()
            policyAction = str(meta_data1['Policy Action:']).lower()
          
            statusMarkDate = str(meta_data1['Date'].tolist()[0]).lower()
            status_reviewer = str(meta_data1['Reviewer'].tolist()[0]).lower()
            status = str(meta_data1['Action Status']).lower()

            commentDate = str(meta_data1['Date'].tolist()[1]).lower()
            comment = str(meta_data1['Comment']).lower()
            comment_reviewer = str(meta_data1['Reviewer'].tolist()[1]).lower()
            meta_data1['From'] = str(meta_data1['From']).lower()
            meta_data1["To"] = str(meta_data1["To"]).lower()
            meta_data1["Cc"] = str(meta_data1["Cc"]).lower()
            participants = meta_data1["From"]+ ";" + meta_data1["To"]+ ";" + meta_data1["Cc"]
            sender = meta_data1["From"]
            if is_not_nan(meta_data1["To"]):
                recipients = meta_data1["To"]
            if is_not_nan(meta_data1["Cc"]):
                recipients = recipients+ ";" + meta_data1["Cc"]

            subject = meta_data1["Subject"]
            messages = tuple(conversation[message_col].tolist())
            df = DataFrame([itemId, messageType, messageDirection, case, captureDate, policyAction, statusMarkDate, status, status_reviewer, commentDate, comment, comment_reviewer, participants, timestamp, language, sender, recipients, subject, conversation, num_of_conversation_turns, messages]).T
            df.columns = ["itemId", "messageType", "messageDirection", "case", "captureDate", "policyAction", "statusMarkDate", "status", "status_reviewer", "commentDate", "comment", "comment_reviewer", "participants", "timestamp", "language", "sender", "recipients", "subject", "conversation", "num_of_conversation_turns", "messages"]

        else:            
            df = DataFrame()
        return df
    
    else:
        text = open(file, 'r').read()
        return text
    


def get_conversation(data):
    length = len(data) - 1
    conversation = data[length]
    conversation.columns = conversation.iloc[0].tolist()
    conversation = conversation.drop(0, axis=0)
    conversation = conversation.reset_index(drop=True)  
    return conversation

def get_redundaunt_info(data):
    data = data[["timestamp", "sender", "recipients", "subject"]]
    data = data.apply(lambda x: " ".join(x), axis=1).value_counts()
    return data

def read_folder(folder, in_type = "html"):
    from os import listdir
    from os.path import join, isfile, isdir
    try:
        files = listdir(folder)
    except:
        files = []
    df = []
    print 
    for file in files:
        file = join(folder, file)
        #print(file)
        if in_type == "html" and isfile(file):
            temp = read_file(file, in_type)
            #print(type(temp))
            df.append(temp)
        elif isdir(file):
            df.append(read_folder(file))
    if len(df)!=0:
        df = concat(df, axis = 0, ignore_index = False)
        df = df.reset_index(drop = True)
    else:
        df = DataFrame()
    return df


def flatten_list_of_list(list_of_list):
    from itertools import chain
    return list(chain.from_iterable(list_of_list))

def clean_sentences(sentences):
    return [clean_strings(string) for string in sentences]

def clean_strings(string):
    from re import sub
    return sub(pattern = "^(nan )*", repl = "", string = string)

def pick_first_language(langs):
    if langs!=None:
        return langs[0]
    else:
        return langdetect.language.Language(lang = "NA", prob = 0)

def is_english_wp_p(langs, p = 0.5):
    return langs.lang == "en" and langs.prob > p

def diffs(index, tokens):
    return [tokens[index], tokens[index+1]]

def merge_words(words):
    return words[0] + words[1]

def correct_tokens(tokens, wrong_corrected, combine_check):
    final_tokens = []
    i = 0
    j = 0
    while i<len(tokens):
        if i!=combine_check[j]:
            final_tokens.append(tokens[i])
        else:
            final_tokens.append(wrong_corrected[combine_check[j]])
            j = j+1
            i = i+1
        i = i+1
    return DataFrame(final_tokens)[0]

def lower(text):
    return text.lower()

def check_spell(row):
    from spellcheck import SpellCheck
    spell_check = SpellCheck('/usr/share/dict/words')
    if len(row[0])==1 or row[1] in [")", "(", "''", "PP$", ",", ":", '``']:
        return row[0]
    else:
        #ret = spell(row[0])
        ret = spell_check.correct(row[0])
        return ret

def is_in_words(word):
    return word in NLP_WORDS

# This is yet to  be developed fully. It currently returns the tokens as they are
def spell_correct_tokens(pos):
    # This only merges 2 consecutive words & checks if they are both incorrectly spelled
    from autocorrect import spell
    try:
        tokens = pos[pos[1]!="SENT"]
        updated_tokens = tokens.apply(check_spell, axis = 1).apply(lower)
        same = updated_tokens != tokens[0]
        diff = DataFrame(same.index.values)[0][same]
        if len(diff)>0:
            wrong = diff.apply(diffs, args = (tokens, ))
            wrong_merge = wrong.apply(merge_words)
            wrong_corrected = wrong_merge.apply(spell).apply(lower)
            same1 = wrong_corrected == wrong_merge
            combine_check = diff[same1]
            wrong_corrected = wrong_corrected[same1]
            same2 = wrong_corrected.apply(is_in_words)
            wrong_corrected = wrong_corrected[same2]
            combine_check = combine_check[same2]
            if len(wrong_corrected)>0:
                tokens = correct_tokens(tokens[0], wrong_corrected, combine_check)
            else:
                tokens = tokens[0]
        else:
            tokens = tokens[0]
        if pos[1][len(pos)-1] == "SENT":
            tokens = tokens.append(DataFrame([pos[0][len(pos)-1]]),
                                   ignore_index=True)
        return tokens.tolist()
    except:
        return pos[0].tolist()

def is_not_none(row):
    return row!=None

def is_not_nan(num):
    try:
        return not(isnan(num))
    except:
        return True

#def spell_correct_pos(pos):
#    try:
#        tokens = spell_correct_tokens(pos)[0].tolist()
#        return tokens
#    except:
#        return pos[0].tolist()
#
def process_NotTag(not_tag):
    text = not_tag.split('"')
    return text[1]

def detect_language(text):
    from langdetect import detect_langs
    try:
        return detect_langs(text)
    except:
        return None

def remove_stopwords(tokens):
    tokens = [token for token in tokens if token not in english_stopwords]
    return tokens

def remove_punctuations(tokens):
    tokens = [token for token in tokens if token not in puncts]
    return tokens

def process_date(date):
    date = date.replace(".", "").split(", ")[1]
    dt = parse(date)
    date = str(dt.year)
    month = str(dt.month)
    day = str(dt.day)
    if len(month) == 1:
        month = "0" + month
    
    if len(day) == 1:
        day = "0" + day
    
    date = date + "/" + month + "/" + day
    return date

def get_maximal_conversation(data, columns):
    max_conv = data[columns].groupby(columns).count().reset_index()
    if 'index' in max_conv.columns:
        max_conv = max_conv.drop(['index'], axis=1)
    
    if 'count' in max_conv.columns:
        max_conv = max_conv.drop(['count'], axis=1)
    
    return max_conv

def filter_data(data, message_col = 'messages'):
    # Retaining only English
    data = data[data['language'] == "en"].reset_index(drop = True)
    data['timestamp'] = data['timestamp'].apply(process_date).reset_index(drop = True)
    # Deduplicating:
    columns = ['participants', 'timestamp', 'sender', 'recipients', message_col]
    max_conv = get_maximal_conversation(data, columns)
    
    max_conv1 = max_conv.merge(max_conv, on = ['participants', 'timestamp', 'sender', 'recipients'], how = 'outer')
    max_conv2 = max_conv1.groupby(['participants', 'timestamp', 'sender', 'recipients', 'messages_x']).count()
    max_conv2 = max_conv2[max_conv2['messages_y']==1].drop(['messages_y'], axis=1)
    shp = max_conv2.shape
    if shp[1]!=0:
        max_conv2.columns = columns
    
    max_conv1 = max_conv1[max_conv1[message_col + '_x'] != max_conv1[message_col + '_y']]
    if max_conv1.shape[0] > 0:
        max_conv1['subset'] = max_conv1.apply(lambda x: set(x[message_col + '_x']).issubset(set(x[message_col + '_y'])), axis=1)
        max_conv1 = max_conv1.drop([message_col + '_y'], axis=1)
        max_conv1.columns = columns + ['subset']
        max_conv1 = max_conv1.groupby(columns).sum().reset_index()
        max_conv1 = max_conv1[max_conv1['subset']==0]
        max_conv3 = max_conv.merge(max_conv1, on = columns, how = 'inner').reset_index(drop = True).drop(['subset'], axis = 1)
        max_conv4 = concat([max_conv2, max_conv3], axis = 0)
        return max_conv4
    else:
        return max_conv2

def filter_senders(data, sender_col = "sender"):
    # Filtering senders with names like "GG *"
    results = data[sender_col].apply(findgg)
    data = data[results == 0]
    return data

def findgg(string):
    return len(findall("gg[\ ]*", string.lower()))

def filter_recipients(data, recipients_col = "recipients"):
    results = data[recipients_col].apply(lambda x: len(x.split(";")))
    data = data[results <= 5]
    return data

def search_patterns(string):
    com = findall(patterns, string.lower())
    return len(com) > 0
