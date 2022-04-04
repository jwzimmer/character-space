from __future__ import annotations

import json
import os

import nltk
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords as sw


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
        config: str | None = None
):
    """
    Generates adjacency data for character names in texts with respect to
    other words. Arguments used by this method and other methods called by it
    are passed in via config, which should be the path to a json file. If no
    config file is passed, the default config file will be used, as specified
    in parameters.
    :param config: See the readme file for details on the config file,
    and see default_config.json for an example.
    :return:
    """
    if config is None:
        config = "default_config.json"
    with open(config, 'r') as file:
        config = json.load(file)

    # Load in the data from the specified input directory
    texts, compounding_dicts, char_names = _load_input_data(config)
    # Tokenize the input texts
    tokenized_texts = _tokenize_texts(config, texts, compounding_dicts)
    # Generate and save embedding data
    _process_texts(config, tokenized_texts, char_names)


def _load_input_data(
        config: dict
) -> (tuple[dict[str, str],
            dict[str, dict[str, str] | None],
            dict[str, list[str]]]):
    """
    Scans the input directory specified in the config file, reading pairs of
    json and text files to create the three basic data structures required
    for tokenization and processing: the text files themselves, lists of
    character names to map, and a set of transformations to be performed on
    the text prior to tokenization.
    :param config: See the readme file for details on the config file.
    :return:
    """
    # Get input directory
    input_directory = config["Input Directory"]

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

    # Verify that we have a 1:1 ratio of json files to text files:
    if len(text_filenames) != len(json_filenames):
        raise ValueError(
            "There are more json files than txt files in the input directory. "
            "Something fishy is going on..."
        )

    # process input files and load data into dictionaries to return
    # The contents of the text files as single strings
    texts = dict()
    # Lists of character names to embed
    char_names = dict()
    # Dictionaries of transformations to perform before tokenizing
    compounding_dicts = dict()

    # Iterate through text files and load their contents into texts,
    # using titles as keys
    for filename in text_filenames:
        filepath = os.path.join(input_directory, filename)
        with open(filepath, 'r') as file:
            text = file.read()
        texts[filename[:-4]] = text.lower()

    # Iterate through json files and load their contents into char_names and
    # compounding_dicts, using titles as keys
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
        if not json_data['char_names']:
            raise ValueError(
                f"No character names were provided for {filename[:-5]}. There "
                f"is nothing for the algorithm to embed."
            )
        char_names[filename[:-5]] = json_data['char_names']

    return texts, compounding_dicts, char_names


def _tokenize_texts(
        config: dict,
        texts: dict[str, str],
        compounding_dicts: dict[str, dict[str, str]]
) -> dict[str, list[str]]:
    """
    Converts the strings in texts into lists of strings comprising words,
    after first performing any transformations specified in compounding_dicts.
    :param config: See the readme file for details on the config file.
    :param texts: A dictionary whose keys are the titles of books, and whose
    values are single strings containing the entire unprocessed text.
    :param compounding_dicts: A dictionary whose keys are the titles of
    books, and whose values are dictionaries. Each of these dictionaries
    describe a transformation to be performed on the text prior to
    tokenizing, with the keys being strings to be replaced, and the values
    being the string to replace them with.
    :return:
    """

    # Create a dictionary to store the results.
    processed_texts = dict()

    # Iterate through each book and process it.
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
            # Exclude empty tokens...
            if cleaned_token != '':
                cleaned_tokenized_text.append(cleaned_token.lower())

        # Optionally Remove stop words
        if config["Remove Stop Words"]:
            # But not if we want to work with the pronouns!
            if config["Process Pronouns"]:
                raise ValueError(
                    "If you opt to remove stop words, then you can't also "
                    "process pronouns. Removal of stopwords also removes "
                    "pronouns."
                )
            stopwords = sw.words('english')
            # The list of nltk stopwords includes some apostrophes, so we'll
            # clean those out...
            cleaned_stopwords = list()
            for word in stopwords:
                cleaned_word = ''.join(c for c in word if c.isalpha())
                cleaned_stopwords.append(cleaned_word)
            cleaned_tokenized_text = (
                [w for w in cleaned_tokenized_text
                 if w not in cleaned_stopwords]
            )

        # We could potentially replace pronouns with the most likely nouns
        # via some algorithm...
        if config["Process Pronouns"]:
            # TODO: Process pronouns - Not implemented at this time.
            pass

        processed_texts[title] = cleaned_tokenized_text

    return processed_texts


def _process_texts(
        config: dict,
        tokenized_texts: dict[str, list[str]],
        char_names: dict[str, list[str]]
):
    """
    Looks for specified character names in the associated text. When a
    character name is found, searches in either direction for words that are
    the correct part of speech (e.g. adjectives, adverbs - specified in
    config file. The distance of these words from the character name are then
    recorded.
    :param config: See the readme file for details on the config file.
    :param tokenized_texts: A dictionary whose keys are titles and values are
    lists of the words of the book in sequence.
    :param char_names: A dictionary whose keys are titles and whose values
    are lists of the character names to embed.
    :return:
    """
    # get dictionary of ttts: tagged tokenized texts
    ttts = {
        title: pos_tag(tokenized_text)
        for title, tokenized_text in tokenized_texts.items()
    }

    # Get parts of speech used to select words of space in which to embed
    # character names.
    pos = config["Included Parts of Speech"]
    if not pos:
        raise ValueError(
            "No parts of speech were specified for embedding."
        )

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

    # Save data
    output_directory = config["Output Directory"]

    # Identify and verify existence of input directory
    if not os.path.exists(output_directory):
        raise ValueError(
            f"Input directory {output_directory} not found."
        )

    # Save results as json file in output directory
    file_path = os.path.join(
        output_directory,
        f"{config['Output File Name']}, "
        f"{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
        f".json"
    )
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4, sort_keys=True, ensure_ascii=False)


if __name__ == '__main__':
    main()
