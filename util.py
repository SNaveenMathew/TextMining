# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:33:25 2017

@author: huijing.deng
"""

from treetaggerwrapper import TreeTagger, make_tags
from math import isnan
#from en_core_web_md import load
from os import environ
from pandas import DataFrame, concat
import langdetect
from autocorrect.nlp_parser import NLP_WORDS
from nltk.corpus import stopwords
from string import punctuation as puncts
from re import findall, sub, compile
from dateparser import parse

treetagger_home = open('treetagger.cfg').read()
environ["TREETAGGER_HOME"] = treetagger_home
tagger = TreeTagger(TAGLANG = 'en')
puncts1 = "[" + puncts + "]"
NLP_WORDS = set([word.lower() for word in NLP_WORDS])
english_stopwords = set(stopwords.words('english'))

# Other utilities:
# 1) Read text file
# 2) Flatten a double-list into list
# 3) Clean beginning of sentences
# 4) Pick language with highest probability from set of languages
# 5) Check whether language is English with prabability > p (default = 0.5)
# 6) Spell correct (in progress)

# Purpose: To run TreeTagger and get the output in TreeTagger format for given text
# Input: String
# Output: List of Tags(word, pos, lemma)
def run_treetagger(text):
    s = tagger.tag_text(text.lower())
    s = make_tags(s)
    return(s)

# Purpose: To read a given FILE of given input type
# Input: File name (path), type of file, message column name (for html_chat)
# Output: Either of:
# 1) csv: DataFrame with same columns as the original file
# 2) excel: DataFrame with same columns as the original file
# 3) html_chat: DataFrame with metadata columns and 'conversation' column with DataFrame of chat history
# and 'messages' column with tuple of all messages in chat
# 4) html_email: DataFrame with 'meta_data' of all emails (From, To, Date, etc.)
# and 'conversation' containing the body of all emails
# 5) If none of the above types, read it as a text file and return string
def read_file(file, in_type = "csv", message_col = "Message"):
    in_type = in_type.lower()
    if in_type == "csv":
        from pandas import read_csv
        return read_csv(file, encoding = "latin1")
    elif in_type == "excel":
        from pandas import read_excel
        return read_excel(file, encoding = "latin1")
    elif in_type == "html_chat":
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
                meta_data[1] = DataFrame(data=d,columns=[0, 1])
            if meta_data[2][0][0] == "No comments have been left":
                d = np.array([["Date","null"],["Comment","null"],["Reviewer","null"]])
                meta_data[2] = DataFrame(data=d,columns=[0, 1])
                
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
            participants = [meta_data1["From"]]
            sender = meta_data1["From"]
            if is_not_nan(meta_data1["To"]):
                recipients = meta_data1["To"]
                participants = participants + [meta_data1["To"]]
            if is_not_nan(meta_data1["Cc"]):
                recipients = recipients+ ";" + meta_data1["Cc"]
                participants = participants + [meta_data1["Cc"]]

            participants.sort()
            participants = tuple(participants)
            subject = meta_data1["Subject"]
            conversation[message_col] = conversation[message_col].apply(remove_punctuations_string).apply(remove_excess_spaces)
            messages = tuple(conversation[message_col].tolist())
            df = DataFrame([itemId, messageType, messageDirection, case, captureDate, policyAction, statusMarkDate, status, status_reviewer, commentDate, comment, comment_reviewer, participants, timestamp, language, sender, recipients, subject, conversation, num_of_conversation_turns, messages]).T
            df.columns = ["itemId", "messageType", "messageDirection", "case", "captureDate", "policyAction", "statusMarkDate", "status", "status_reviewer", "commentDate", "comment", "comment_reviewer", "participants", "timestamp", "language", "sender", "recipients", "subject", "conversation", "num_of_conversation_turns", "messages"]

        else:
            df = DataFrame()
        return df
    elif in_type == "html_email":
        from bs4 import BeautifulSoup
        from pandas import read_html
        from dateparser import parse
        html = BeautifulSoup(open(file, "rb").read(), "html.parser")
        all_fields = ["From ", "Date ", "To", "Cc", "Subject"]
        all_fields_pattern = "|".join(all_fields)
        readhtml = read_html(file)
        dic = process_meta_data(" ".join(readhtml[0].T[0].tolist()), all_fields_pattern)
        t1 = readhtml[1]
        values = t1[1].tolist()
        keys = t1[0]
        for i in range(len(keys)):
            dic[keys[i]] = values[i]
        
        meta_data = [dic]
        tex = [a.text for a in html.findAll("p", class_="MsoNormal") if a.text!='\xa0']
        all_content = get_all_email_content(tex)
        all_fields = ["From: ", "Sent: ", "To: ", "Subject: "]
        all_fields_pattern = "|".join(all_fields)
        metadata_start_pattern = "^[\>]*[\ ]*From: "
        metadata_stop_pattern = "Subject: "
        contents, meta_data = get_contents_meta_data(all_content, all_fields_pattern, metadata_start_pattern, metadata_stop_pattern, in_type, meta_data)
        df = DataFrame({"meta_data": meta_data, "conversation": contents})
        return df
    elif in_type.lower() == "enron_email":
        try:
            all_content = open(file, 'r').readlines()
            all_fields = ["From: ", "Sent: ", "To: ", "Subject: ", "Message\-ID: ", "Date: ",
            "Mime\-Version: ", "Content\-Type: ", "Content\-Transfer\-Encoding: ", "X\-From: ",
            "X\-To: ", "X\-cc: ", "X\-bcc: ", "X\-Folder: ", "X\-Origin: ", "X\-FileName: ", "Subject:\t"]
            all_fields_pattern = "|".join(all_fields)
            metadata_start_pattern = "^[\>]*[\ ]*From: |^[\>]*[\ ]*Message\-ID: "
            metadata_stop_pattern = "^[\>]*[\ ]*Subject:[\t]*[\ ]*|^[\>]*[\ ]*X\-FileName: "
            contents, meta_data = get_contents_meta_data(all_content, all_fields_pattern, metadata_start_pattern, metadata_stop_pattern, in_type)
            df = DataFrame({"meta_data": meta_data, "conversation": contents})
        except:
            df = DataFrame()
        return df
    else:
        text = open(file, 'r').read()
        return text


# Purpose: To get the list of contents of email from a list of strings
# Input: List of strings
# Output: List of conversations
def get_all_email_content(tex):
    all_content = [sub(string = a, pattern = "[\-]*Original Message[\-]*", repl = "").strip() for a in tex]
    return all_content

def get_contents_meta_data(all_content, all_fields_pattern, metadata_start_pattern, metadata_stop_pattern, in_type, meta_data = []):
    start_index = [i for i, content in enumerate(all_content) if len(findall(string = content, pattern = metadata_start_pattern))>0]
    stop_index = [i for i, content in enumerate(all_content) if len(findall(string = content, pattern = metadata_stop_pattern))>0]
    if in_type.lower() == "enron_email":
        start_index = start_index[1:]
        stop_index = stop_index[1:]
    
    start_index = start_index + [len(all_content)]
    stop_index = [-1] + stop_index
    ranges = [(stop_index[i]+1, start_index[i]) for i, val in enumerate(start_index)]
    if in_type.lower() == "enron_email":
        ranges = ranges[1:]
    
    contents = []
    for rng in ranges:
        string = "\n".join(all_content[rng[0]:rng[1]])
        contents.append(string)
    
    start_index = start_index[:-1]
    stop_index = stop_index[1:]
    if in_type.lower() == "enron_email":
        start_index[0] = 0
    
    ranges = [(start_index[i], stop_index[i]+1) for i, val in enumerate(start_index)]
    meta_d = []
    for rng in ranges:
        string = (". \n".join([a for a in all_content[rng[0]:rng[1]] if a!= ""])).strip()
        meta_d.append(string)
    
    for meta in meta_d:
        meta_data.append(process_meta_data(meta, all_fields_pattern))
    
    return contents, meta_data

def process_meta_data(meta_data_string, all_fields_pattern):
    from re import split, findall
    keys = [sub(string = st, pattern = "[^A-Za-z0-9]", repl = "") for st in findall(string = meta_data_string, pattern = all_fields_pattern)]
    vals = [st.strip() for st in split(string = meta_data_string, pattern = all_fields_pattern)[1:]]
    dic = {}
    for i in range(len(vals)):
        dic[keys[i]] = vals[i]
    
    return dic

# Purpose: To remove punctuations from a string
# Input: String
# Output: String
def remove_punctuations_string(string):
    return sub(pattern = puncts1, repl = "", string = string)

# Purpose: To convert >=2 spaces into 1 space in a string
# Input: String
# Output: String
def remove_excess_spaces(string):
    return sub(pattern = " {2,}", repl = " ", string = string)

def get_conversation(data):
    length = len(data) - 1
    conversation = data[length]
    conversation.columns = conversation.iloc[0].tolist()
    conversation = conversation.drop(0, axis=0)
    conversation = conversation.reset_index(drop=True)  
    return conversation

# Purpose: To remove redundant data points
# Input: DataFrame with columns ["timestamp" (date), "sender", "recipients", "subject", ...]
# Output: DataFrame with counts of unique ["timestamp" (date), "sender", "recipients", "subject"]
def get_redundaunt_info(data):
    data = data[["timestamp", "sender", "recipients", "subject"]]
    data = data.apply(lambda x: " ".join(x), axis=1).value_counts()
    return data

# Purpose: To recursively read different files in a folder (only 1 type of file per folder)
# Input: Folder name
# Output: DataFrame with all row-binded read_file results
def read_folder(folder, in_type = "html_chat"):
    from os import listdir
    from os.path import join, isfile, isdir
    in_type = in_type.lower()
    try:
        files = listdir(folder)
    except:
        files = []
    df = []
    for file in files:
        file = folder + "/" + file
        #print(file)
        if isdir(file):
            df.append(read_folder(file))
        elif in_type == "html_chat" or in_type == "html_email" or in_type == "enron_email" and isfile(file):
            temp = read_file(file, in_type)
            #print(type(temp))
            df.append(temp)
    if len(df)!=0:
        df = concat(df, axis = 0, ignore_index = False)
        df = df.reset_index(drop = True)
    else:
        df = DataFrame()
    return df


# Purpose: To flatten list of list of ... into linear list
# Input: List of list of ...
# Output: List (flattened completely)
def flatten_list_of_list(list_of_list):
    from itertools import chain
    return list(chain.from_iterable(list_of_list))

# Purpose: To clean a list of sentences
# Input: List of strings
# Output: List of strings
def clean_sentences(sentences):
    return [clean_strings(string) for string in sentences]

# Purpose: To clean a sentence
# Input: String
# Output: String
def clean_strings(string):
    return sub(pattern = "^(nan )*", repl = "", string = string)

# Purpose: To pick the first language from output of language detection
# Input: Languages (list of strings with probability)
# Output: First language (string - highest probability)
def pick_first_language(langs):
    if langs!=None:
        return langs[0]
    else:
        return langdetect.language.Language(lang = "NA", prob = 0)

# Purpose: To choose English if probability > threshold
# Input: List of languages
# Output: Boolean (True / False)
def is_english_wp_p(langs, p = 0.5):
    return langs.lang == "en" and langs.prob > p

# Purpose: To compute list of tokens and lag 1 of tokens
# Input: index and tokens
# Output: tokens and lag1(tokens)
def diffs(index, tokens):
    return [tokens[index], tokens[index+1]]

# Purpose: To combine 2 words in a list
# Input: List of 2 strings
# Output: String
def merge_words(words):
    return words[0] + words[1]

# Purpose: To combine 2 words if both words are incorrctly spelled and combination is correct
# Input: Tokens with spell check, combined words with spelling checked
# Output: DataFrame of combined tokens
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

# Purpose: To convert string to lower case
# Input: String
# Output: String
def lower(text):
    return text.lower()

# Purpose: To check spelling of tags
# Input: Tags
# Output: Spelling corrected tags
def check_spell(row):
    from spellcheck import SpellCheck
    spell_check = SpellCheck('/usr/share/dict/words')
    if len(row[0])==1 or row[1] in [")", "(", "''", "PP$", ",", ":", '``']:
        return row[0]
    else:
        #ret = spell(row[0])
        ret = spell_check.correct(row[0])
        return ret

# Purpose: To check whether word is in predefined set of words
# Input: Word
# Output: Boolean
def is_in_words(word):
    return word in NLP_WORDS

# Purpose: To combine 2 words if both words are incorrctly spelled and combination is correct
# Input: POS DataFrame
# Output: DataFrame of combined tokens
def spell_correct_tokens(pos):
    # This only merges 2 consecutive words & checks if they are both incorrectly spelled
    from autocorrect import spell
    pos = DataFrame(pos)
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

# Purpose: To check whether row is None
# Input: Row
# Output: Boolean
def is_not_none(row):
    return row!=None

# Purpose: To check whether number is NaN
# Input: Number
# Output: Boolean
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

# Purpose: To process Tags that are not well formed tags
# Input: Tag
# Output: String
def process_NotTag(not_tag):
    text = not_tag.split('"')
    return text[1]

# Purpose: To detect language of a strong
# Input: String
# Output: List of languages
def detect_language(text):
    from langdetect import detect_langs
    try:
        return detect_langs(text)
    except:
        return None

# Purpose: To remove stop words
# Input: List of tokens
# Output: List (of words without stopwords)
def remove_stopwords(tokens):
    tokens = [token for token in tokens if token not in english_stopwords]
    return tokens

# Purpose: To remove punctuations
# Input: List of tokens
# Output: List (of words without punctuations)
def remove_punctuations(tokens):
    tokens = [token for token in tokens if token not in puncts]
    return tokens

# Purpose: To convert date string to date format
# Input: Date string
# Output: Date
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

# Purpose: To get conversation of max length
# Input: DataFrame with conversation column
# Output: Deduplicated conversation
def get_maximal_conversation(data, columns):
    max_conv = data[columns].groupby(columns).count().reset_index()
    if 'index' in max_conv.columns:
        max_conv = max_conv.drop(['index'], axis=1)
    
    if 'count' in max_conv.columns:
        max_conv = max_conv.drop(['count'], axis=1)
    
    return max_conv

# Purpose: To deduplicate a DataFrame of conversation
# Input: DataFrame with message column
# Output: DataFrame (after removing duplicates)
def filter_data(data, message_col = 'messages'):
    # Retaining only English
    data = data[data['language'] == "en"].reset_index(drop = True)
    data['timestamp'] = data['timestamp'].apply(process_date).reset_index(drop = True)
    # Deduplicating:
    columns = ['participants', 'timestamp', message_col]
    max_conv = get_maximal_conversation(data, columns)
    
    max_conv1 = max_conv.merge(max_conv, on = ['participants', 'timestamp'], how = 'outer')
    max_conv2 = max_conv1.groupby(['participants', 'timestamp', 'messages_x']).count()
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

# Purpose: To filter conversations and remove conversations with sender like GG *
# Input: DataFrame of conversations
# Output: DataFrame (after filtering senders)
def filter_senders(data, sender_col = "sender"):
    # Filtering senders with names like "GG *"
    results = data[sender_col].apply(findgg)
    data = data[results == 0]
    return data

# Purpose: To check whether sender is of GG * format
# Input: String of senders
# Output: Length of GG * pattern
def findgg(string):
    return len(findall("gg[\ ]*", string.lower()))

# Purpose: To remove spurious recipients
# Input: DataFrame of conversations
# Output: DataFrame (after removing spurious recipients)
def filter_recipients(data, recipients_col = "recipients"):
    results = data[recipients_col].apply(lambda x: len(x.split(";")))
    data = data[results <= 5]
    return data

# Purpose: To search for a pattern in given string
# Input: String and pattern
# Output: Boolean
def search_pattern(string, pattern):
    com = findall(pattern, string.lower())
    return len(com) > 0

# Purpose: To search for patterns in given string
# Input: String and pattern
# Output: Boolean
def search_patterns(string, patterns):
    results = patterns.apply(lambda x: search_pattern(x, string))
    return results

# Purpose: To calculate semantic similarity of words in word2vec
# Input: word2vec model
# Output: m x m similarity matrix
def get_semantic_similarity(word2vec_model):
    from sklearn.metrics.pairwise import cosine_similarity
    mat = word2vec_model[word2vec_model.wv.vocab]
    sim = DataFrame(cosine_similarity(mat))
    sim.columns = word2vec_model.wv.vocab
    sim.index = word2vec_model.wv.vocab
    return sim

# Purpose: To calculate fuzzy similarity of words in vocab
# Input: Vocabulary, type of fuzzy similarity
# Output: m x m similarity matrix
def get_character_similarity(vocab, ratio_type = 'ratio'):
    from fuzzywuzzy import fuzz
    vocab = DataFrame(vocab)
    vocab['dummy'] = 1
    vocab = vocab.merge(vocab, on = 'dummy', how = 'outer')
    vocab = vocab.drop(['dummy'], axis = 1)
    vocab.columns = ['word1', 'word2']
    if ratio_type == "ratio":
        func = fuzz.ratio
    elif ratio_type == "partial_ratio":
        func = fuzz.partial_ratio
    elif ratio_type == "token_sort_ratio":
        func = fuzz.token_sort_ratio
    else:
        func = fuzz.token_set_ratio
    vocab[ratio_type] = vocab.apply(lambda x: (func(x['word1'], x['word2']))/100, axis=1).to_frame()
    vocab = vocab.pivot_table(index = ['word1'], columns = ['word2'])
    del vocab.index.name
    vocab.columns = vocab.columns.droplevel()
    return vocab

def get_word_lda_topics(word):
    try:
        return (word, lda_model.get_term_topics(word))
    except:
        return None
