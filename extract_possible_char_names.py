#pip install -U textblob

from textblob import TextBlob
import re
import pandas as pd
from collections import Counter
import json

def pad_punctuation(text):
  punc_list = ["!","&",".","?",",","-","——","(",")","~","—",'"',"_",":",";"]
  words = text.lower()
  for punc in punc_list:
    words = re.sub(re.escape(punc)," "+punc+" ",words)
  words = words.split()
  punc_from_text = [x for x in words if x in punc_list]
  return words, punc_from_text

# Open the text file for the book
text = open("prideandprejudice.txt")
text = text.read()
text = text.lower()

blob = TextBlob(text)

tag_dict = {}
for tup in blob.tags:
  tag_dict[tup[0]] = tup[1]

# Load whatever dictionaries you want to use
labmt_word_df = pd.read_csv("labmt.txt",sep="\t")
pds_word_df = pd.read_csv("http://pdodds.w3.uvm.edu/permanent-share/ousiometry_data.txt", sep="\t")
all_words = set(list(pds_word_df["word"])).union(set(list(labmt_word_df["word"])))

possible_names = []
prefixes = ["mr. ", "mrs. ", "miss ", "mister ", "ms. ", "madame ", "lady ", "sir "]
for n in pap_blob.noun_phrases:
  for prefix in prefixes:
    if prefix in n:
      pattern = re.compile(r'\b'+prefix+'[a-z]*[\s[a-z]*]*')
      match = re.search(pattern, n)
      #print(n)
      possible_names.append(match.group())
    else: pass
  new_word = ""
  for word in n.split():
    if word in tag_dict.keys():
      if tag_dict[word] == "NN":
        if word not in all_words:
          new_word += " " + word
    else:
      pass
  #print(new_word)
  possible_names.append(new_word)

name_counts = Counter(possible_names)

possible_names2 = []
for word in name_counts:
  if name_counts[word] < 2: pass
  else: possible_names2.append(word)

big_dict = {"char_names":[]}
for name in possible_words2:
  big_dict["char_names"].append(name)

compounding_dict = {}
for name in possible_words2:
  compounding_dict[name] = name

big_dict["compounding_dict"] = compounding_dict

with open('big_dict.json', 'w') as f:
    json.dump(big_dict, f)