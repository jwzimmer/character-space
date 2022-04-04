from __future__ import annotations

from textblob import TextBlob
import re
from collections import Counter
import json
import os


def main():
    process_directory(None)


def process_directory(
        config: str | None = None
):
    """
    For every text in the input directory specified in the config file,
    creates a json file that lists character names in the text and contains
    the skeleton of a map for transforming each identified name. Files will
    be saved in the specified output directory and have the same name as the
    input text file but with a .json suffix.
    The created json files are crude and should be manually edited. This code
    is intended to save time in the construction of character lists / maps,
    not to entirely replace necessary human input.
    :param config: See the readme file for details on the config file,
    and see default_config.json for an example.
    :return:
    """

    if config is None:
        config = "./default_config.json"
    with open(config, 'r') as file:
        config = json.load(file)

    input_dir = config["Input Directory"]
    output_dir = config["Output Directory"]
    _iterate_through_text_files(input_dir, output_dir)


def _iterate_through_text_files(input_dir, output_dir):
    directory = os.fsencode(input_dir)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        text = _open_book_text_file(
            os.path.join(input_dir, filename)
        )
        _make_char_name_dict(text, output_dir, filename)


def _pad_punctuation(text):
    punc_list = ["!", "&", ".", "?", ",", "-", "——", "(", ")", "~", "—", '"',
                 "_", ":", ";"]
    words = text.lower()
    for punc in punc_list:
        words = re.sub(re.escape(punc), " " + punc + " ", words)
    words = words.split()
    punc_from_text = [x for x in words if x in punc_list]
    return words, punc_from_text


# Open the text file for the book
def _open_book_text_file(filename):
    text = open(filename)
    text = text.read()
    return text


def _make_char_name_dict(text, output_dir, filename):
    uppercase_pattern = re.compile(
        r'(?<![\.|\!|\?]\s)(?<!")(?<!\n)[A-Z]+[a-z]+', re.MULTILINE)
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
            # print(i)
            try:
                all_words.append(line.rstrip().lower())
            except Exception as e:
                print('******************')
                print('Found error')
                print(e)
                print('******************')
        all_words = set(all_words)

    ignore_words = ["the", "after", "before", "in", "on", "of", "herein",
                    "whereby", "your", "my", "mine", "yours",
                    "his", "her", "hers", "their", "theirs", "there", "here",
                    "him", "he", "she", "they", "thine",
                    "afore", "near", "next", "to", "across", "between", "below",
                    "above", "beneath", "under", "this",
                    "that", "those", "them", "behaviour", "colour",
                    "neighbourhood", "whilst", "and", "as", "at", "if", "but",
                    "by",
                    "do", "every", "from", "had", "how", "humour", "it", "yes",
                    "no", "oh", "with", "you", "then",
                    "we", "what", "when", "while", "st", "sir", "house", "park",
                    "madam", "ma", "mr", "mrs", "miss",
                    "mister", "ms", "madame", "lady", "doctor", "dr",
                    "reverend", "rev", "lieutenant", "lt", "colonel",
                    "col", "captain", "missus", "master", "mistress", "hon",
                    "honorable", "general", "hill",
                    "monday", "tuesday", "wednesday", "thursday", "friday",
                    "saturday", "sunday",
                    "january", "february", "march", "april", "may", "june",
                    "july", "august", "september", "october",
                    "november", "december",
                    "i", "ii", "iii", "iv",
                    "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii",
                    "xiv", "xv", "xvi", "xvii", "xviii",
                    "xix", "xx", "xxi", "xxii", "xxiii", "xxiv", "xxv"]
    ignore_words = set(ignore_words)
    all_words = all_words.union(ignore_words)

    possible_names = [x for x in uppercase_names if x not in ignore_words]
    prefixes = ["mr. ", "mrs. ", "miss ", "mister ", "ms. ", "madame ", "lady ",
                "sir ", "doctor ", "dr. ",
                "reverend ", "rev. ", "lieutenant ", "lt. ", "colonel ",
                "col. ", "captain ", "missus ",
                "master ", "mistress ", "hon. ", "honorable ", "general "]
    for n in blob.noun_phrases:
        for prefix in prefixes:
            if prefix in n:
                pattern = re.compile(r'\b' + prefix + '[a-z]*[\s*[a-z]*]*')
                match = re.search(pattern, n)
                # print(n)
                try:
                    if match.group()[-1] == " ":
                        pass
                    else:
                        possible_names.append(match.group())
                except:
                    pass
            else:
                pass
        new_word = ""
        for word in n.split():
            if word in tag_dict.keys():
                if tag_dict[word] == "NN":
                    if word not in all_words:
                        new_word += " " + word
            else:
                pass
        # print(new_word)
        if new_word != "":
            possible_names.append(new_word[1:])

    name_counts = Counter(possible_names)

    possible_names2 = []
    for word in name_counts:
        if name_counts[word] < 5:
            pass
        else:
            possible_names2.append(word)

    big_dict = {"char_names": []}
    for name in sorted(possible_names2):
        big_dict["char_names"].append(name)

    compounding_dict = {}
    for name in sorted(possible_names2):
        compounding_dict[name] = name

    big_dict["compounding_dict"] = compounding_dict

    file_path = os.path.join(
        output_dir, filename[:-4] + '.json'
    )
    with open(file_path, 'w') as f:
        json.dump(big_dict, f, indent=4, sort_keys=True, ensure_ascii=False)


if __name__ == '__main__':
    main()
