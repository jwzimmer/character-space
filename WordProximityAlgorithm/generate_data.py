from __future__ import annotations

import json
import os
import re

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords as sw

import parameters as p

# You can comment out the below after you've run the code once. It saves
# files the nltk library needs in an nltk specific directory in your home
# directory. If you don't like having that directory there, you can just
# delete it after you've used the nltk library to run some code and it won't
# cause any problems.
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('tagsets')
nltk.download('wordnet')


def main():
    generate_data(None)


def generate_data(
        config: dict | None
):
    if config is None:
        config = p.DEFAULT_CONFIG_FILE

    texts, compounding_dicts, char_names = load_input_data(config)
    tokenized_texts = tokenize_texts(config, texts, compounding_dicts)
    process_texts(config, tokenized_texts, char_names)


def load_input_data(
        config: dict
) -> tuple[dict[str, str], dict[str, dict[str, str] | None], dict[str, list[str]]]:
    # Get input directory
    input_directory = config["Input Directory"]
    if input_directory is None:
        input_directory = p.INPUT_DIR

    # Identify and verify existence of input directory
    if not os.path.exists(input_directory):
        raise ValueError(
            f"Input directory {input_directory} not found."
        )

    # Get filenames from input directory
    text_filenames = list()
    json_filenames = list()
    for filename in os.listdir(input_directory):
        if filename[-4:] == '.txt':
            text_filenames.append(filename)
        elif filename[-5:] == '.json':
            json_filenames.append(filename)
        else:
            raise ValueError(
                f"Found file {filename} in input directory. Directory should "
                f"contain only txt and json files."
            )

    # Verify that there were files in directory...
    if not text_filenames:
        raise ValueError(
            f"No text files were found in {input_directory}."
        )

    # Verify that there is a json file for every text file
    for filename in text_filenames:
        if (filename[:-4] + '.json') not in json_filenames:
            raise ValueError(
                f"File {filename} in input directory has no associated json "
                f"file."
            )

    # process input files and load data into dictionaries to return
    # The contents of the text files as single strings
    texts = dict()
    # Lists of character names to embed
    char_names = dict()
    # Dictionaries of transformations to perform before tokenizing
    compounding_dicts = dict()

    for filename in text_filenames:
        filepath = os.path.join(input_directory, filename)
        with open(filepath, 'r') as file:
            text = file.read()
        texts[filename[:-4]] = text.lower()

    for filename in json_filenames:
        filepath = os.path.join(input_directory, filename)
        with open(filepath, 'r') as file:
            json_data = json.load(file)

        # Get compounding dictionary
        if 'compounding_dict' not in json_data:
            raise ValueError(
                f"'compounding_dict' not found in {filename}"
            )
        # Type None is used to indicate that no compounding should be
        # performed
        if json_data['compounding_dict'] is None:
            compounding_dicts[filename[:-5]] = None
        # Otherwise, should be a dictionary with strings for keys and values
        else:
            if type(json_data['compounding_dict']) is not dict:
                raise ValueError(
                    f"'compounding_dict' in {filename} should be None or of "
                    f"type dict."
                )
            for key, value in json_data['compounding_dict'].items():
                if type(key) is not str:
                    raise ValueError(
                        f"Key {key} in compounding dictionary in file "
                        f"{filename} is not of type string."
                    )
                if type(value) is not str:
                    raise ValueError(
                        f"Value {value} in compounding dictionary in file "
                        f"{filename} is not of type string."
                    )
            compounding_dicts[filename[:-5]] = json_data['compounding_dict']

        # Get list of character names
        if 'char_names' not in json_data:
            raise ValueError(
                f"'char_names' not found in {filename}"
            )
        if type(json_data['char_names']) is not list:
            raise ValueError(
                f"'char_names' in {filename} should be of type list."
            )
        for char_name in json_data['char_names']:
            if type(char_name) is not str:
                raise ValueError(
                    f"Type of character name {char_name} in file {filename} is "
                    f"not string."
                )
        char_names[filename[:-5]] = json_data['char_names']

    return texts, compounding_dicts, char_names


def tokenize_texts(
        config: dict,
        texts: dict[str, str],
        compounding_dicts: dict[str, dict[str, str]]
) -> dict[str, list[str]]:

    processed_texts = dict()

    for title, text in texts.items():

        # Make all replacements specified in compounding dictionary for text
        if compounding_dicts[title] is not None:
            for current, replacement in compounding_dicts[title].items():
                text = text.replace(current, replacement)

        # Tokenize Text
        tokenized_text = text.split()

        # Remove non-alphanumeric characters and convert to lower case
        cleaned_tokenized_text = list()
        for token in tokenized_text:
            cleaned_token = ''.join(c for c in token if c.isalpha())
            if cleaned_token != '':
                cleaned_tokenized_text.append(cleaned_token.lower())

        # Remove stop words
        if config["Remove Stop Words"]:
            if config["Process Pronouns"]:
                raise ValueError(
                    "If you opt to remove stop words, then you can't also "
                    "process pronouns. Removal of stopwords also removes "
                    "pronouns."
                )
            stopwords = sw.words('english')
            cleaned_stopwords = list()
            for word in stopwords:
                cleaned_word = ''.join(c for c in word if c.isalpha())
                cleaned_stopwords.append(cleaned_word)
            cleaned_tokenized_text = (
                [w for w in cleaned_tokenized_text
                 if w not in cleaned_stopwords]
            )

        if config["Process Pronouns"]:
            # TODO: Process pronouns - come back to this later.
            pass

        processed_texts[title] = cleaned_tokenized_text

    return processed_texts


def process_texts(
        config: dict,
        tokenized_texts: dict[str, list[str]],
        char_names: dict[str, list[str]]
):
    # get dictionary of ttts: tagged tokenized texts
    ttts = {
        title: pos_tag(tokenized_text)
        for title, tokenized_text in tokenized_texts.items()
    }

    pos = config["Included Parts of Speech"]

    # Create list of all words whose relationships to characters we will
    # collect data on.
    neighbors = list()
    for title, ttt in ttts.items():
        neighbors += [
            tagged_word[0] for tagged_word in ttt
            if (tagged_word[1] in pos)
        ]
    neighbors = list(set(neighbors))

    # Get list of all character names in all texts:
    all_char_names = list()
    for title, names in char_names.items():
        all_char_names += names
    all_char_names = list(set(all_char_names))

    # Create data structure to store proximity of neighbors to character names
    data = {
        name: {neighbor: list() for neighbor in neighbors}
        for name in all_char_names
    }

    # Get maximum distance forwards and backwards to search
    window = config["Proximity Window"]

    for title, ttt in ttts.items():
        # We will not check the relationship of other character names to our
        # targets for embedding, and we will see if our configuration file
        # includes other words to not embed.
        excluded_words = char_names[title] + config["Words to Exclude"]

        for i, word in enumerate(ttt):
            # See if the word is in our local list of names. If not,
            # we're done with this word
            if word[0] not in char_names[title]:
                continue

            # Find the first and last index of the window
            i_first = i - window
            if i_first < 0:
                i_first = 0
            i_last = i + window
            if i_last >= len(ttt):
                i_last = len(ttt)-1

            # Update data with the distance of neighbor words in the window
            # from the character name, if they are an appropriate pos
            for j in list(range(i_first, i)) + list(range(i, i_last+1)):
                if ttt[j][1] in pos and ttt[j][0] not in excluded_words:
                    data[word[0]][ttt[j][0]].append(
                        (window - (abs(i-j)-1))
                    )

    pass






if __name__ == '__main__':
    main(p.DEFAULT_CONFIG_FILE)
