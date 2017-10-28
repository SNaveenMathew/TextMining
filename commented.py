# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:38:49 2017

@author: naveen.nathan
"""

#from collections import Counter
#from enchant import Dict, check, suggest
#from re import findall

#def P(word, N=sum(WORDS.values())):
#    "Probability of `word`."
#    return WORDS[word] / N
#
#def correction(word): 
#    "Most probable spelling correction for word."
#    return max(candidates(word), key=P)
#
#def candidates(word): 
#    "Generate possible spelling corrections for word."
#    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])
#
#def known(words): 
#    "The subset of `words` that appear in the dictionary of WORDS."
#    return set(w for w in words if w in WORDS)
#
#def edits1(word):
#    "All edits that are one edit away from `word`."
#    letters    = 'abcdefghijklmnopqrstuvwxyz'
#    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
#    deletes    = [L + R[1:]               for L, R in splits if R]
#    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
#    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
#    inserts    = [L + c + R               for L, R in splits for c in letters]
#    return set(deletes + transposes + replaces + inserts)
#
#def edits2(word): 
#    "All edits that are two edits away from `word`."
#    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
#
#d = Dict("en_US")


#def tokenize_sentence_nltk(text, get_lemma = False, delim_list = ["!", "?"]):
#    text = text.replace("\n", ". ").replace("\xa0", " ")
##    for delim in delim_list:
##        text = [sent.split(delim) for sent in text if sent != "" and sent != ' ']
##        text = flatten_list_of_list(text)
#    text = sent_tokenize(text)
#    return(text)

#for i, item in enumerate(first_language):
#    # Running the preprocessing only if language is English with probability > 0.5
#    if(item.prob > 0.5 and item.lang == "en"):
#        sentences = tokenize_sentence_nltk(strings[i])
#        words = [tokenize_stanford(sentence) for sentence in sentences]
#        words = [tokenize_treetagger(sentence) for sentence in sentences]
#        #pos = run_stanford_pos_tag(tokens = words, in_type = "tokens")
#        #pos = run_treetagger_pos_tag(text = strings[i])
#        #for i, word in enumerate(words):
#        #    words[i] = spell(words)
#        #    print(words[i])
#        print(words)
#        #print(pos)
#        # do something here
#    else:
#        print(0)
#        # leave the text unprocessed for now

#def bigram_model(unigrams):
#    phrases = Phrases(unigrams)
#    bigram = Phraser(phrases)
#    return(bigram)

#def separate_token_pos_sentence(sentence_token_pos):
#    sentence_token_pos = DataFrame(sentence_token_pos)
#    token = sentence_token_pos[0].tolist()
#    pos = sentence_token_pos[1].tolist()
#    return(token, pos)
#

#def spell_correct_sentences(pos):
#    tokens = pos.apply(spell_correct_tokens)
#    return(tokens)
#
