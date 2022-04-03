#pip3 install -U textblob

from textblob import TextBlob
import re
import pandas as pd
from collections import Counter
import json
import os

def pad_punctuation(text):
  punc_list = ["!","&",".","?",",","-","——","(",")","~","—",'"',"_",":",";"]
  words = text.lower()
  for punc in punc_list:
    words = re.sub(re.escape(punc)," "+punc+" ",words)
  words = words.split()
  punc_from_text = [x for x in words if x in punc_list]
  return words, punc_from_text

# Open the text file for the book
def open_book_text_file(filename):
  text = open(filename)
  text = text.read()
  return text

def iterate_through_text_files(directory_in_str):
  directory = os.fsencode(directory_in_str)
  for file in os.listdir(directory):
    filename = os.fsdecode(file)
    text = open_book_text_file(
      os.path.join(directory_in_str, filename)
    )
    make_char_name_dict(text, filename)

def make_char_name_dict(text, filename):
  uppercase_pattern = re.compile(r'(?<![\.|\!|\?]\s)(?<!")(?<!\n)[A-Z]+[a-z]+', re.MULTILINE)
  uppercase_names = re.findall(uppercase_pattern, text)
  uppercase_names = [x.lower() for x in uppercase_names]

  text = text.lower()

  blob = TextBlob(text)

  tag_dict = {}
  for tup in blob.tags:
    tag_dict[tup[0]] = tup[1]

  with open("parsed_dict.txt") as file:
    all_words = []
    for i, line in enumerate(file):
      #print(i)
      try:
        all_words.append(line.rstrip().lower())
      except Exception as e:
        print('******************')
        print('Found error')
        print(e)
        print('******************')
    all_words = set(all_words)

  ignore_words = ["the","after","before","in","on","of","herein","whereby","your","my","mine","yours",
                  "his","her","hers","their","theirs","there","here","him","he","she","they","thine",
                  "afore","near","next","to","across","between","below","above","beneath","under","this",
                  "that","those","them","behaviour","colour","neighbourhood","whilst","and","as","at","if","but","by",
                  "do","every","from","had","how","humour","it","yes","no","oh","with","you","then",
                  "we","what","when","while","st","sir","house","park","madam","ma","mr","mrs","miss",
                  "mister","ms","madame","lady","doctor","dr","reverend","rev","lieutenant","lt","colonel",
                  "col","captain","missus","master","mistress","hon","honorable","general","hill",
                  "monday","tuesday","wednesday","thursday","friday","saturday","sunday",
                  "january","february","march","april","may","june","july","august","september","october",
                  "november","december",
                  "i","ii","iii","iv",
                  "v","vi","vii","viii","ix","x","xi","xii","xiii","xiv","xv","xvi","xvii","xviii",
                  "xix","xx","xxi","xxii","xxiii","xxiv","xxv"]
  ignore_words = set(ignore_words)
  all_words = all_words.union(ignore_words)

  possible_names = [x for x in uppercase_names if x not in ignore_words]
  prefixes = ["mr. ", "mrs. ", "miss ", "mister ", "ms. ", "madame ", "lady ", "sir ", "doctor ", "dr. ",
              "reverend ", "rev. ", "lieutenant ", "lt. ", "colonel ", "col. ", "captain ", "missus ",
              "master ", "mistress ", "hon. ", "honorable ", "general "]
  for n in blob.noun_phrases:
    for prefix in prefixes:
      if prefix in n:
        pattern = re.compile(r'\b'+prefix+'[a-z]*[\s*[a-z]*]*')
        match = re.search(pattern, n)
        #print(n)
        try:
          if match.group()[-1] == " ":
            pass
          else:
            possible_names.append(match.group())
        except: pass
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
    if new_word != "":
      possible_names.append(new_word[1:])

  name_counts = Counter(possible_names)

  possible_names2 = []
  for word in name_counts:
    if name_counts[word] < 5: pass
    else: possible_names2.append(word)

  big_dict = {"char_names":[]}
  for name in sorted(possible_names2):
    big_dict["char_names"].append(name)

  compounding_dict = {}
  for name in sorted(possible_names2):
    compounding_dict[name] = name

  big_dict["compounding_dict"] = compounding_dict

  with open(filename[:-4]+'.json', 'w') as f:
      json.dump(big_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

if __name__ == '__main__':
  dirname = "/home/denis/PycharmProjects/character-space/WordProximityAlgorithm/Example/example_input"
  #dirname = "/Users/jzimmer1/Documents/GitHub/character-space/TestDirectory"
  iterate_through_text_files(dirname)