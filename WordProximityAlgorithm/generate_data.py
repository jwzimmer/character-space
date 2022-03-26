from __future__ import annotations

import json
import os

import nltk

import parameters as p

nltk.download('punkt')


def main(config_file_path):
    with open(config_file_path, 'r') as file:
        config = json.load(file)

    generate_data(config)


def generate_data(
        config: dict
):
    texts, compounding_dicts, char_names = load_input_data(config)
    processed_texts = preprocess_texts(config, texts, compounding_dicts)
    tokenized_texts = tokenize_texts(config, processed_texts)
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
        texts[filename[:-4]] = text

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


def preprocess_texts(
        config: dict,
        texts: dict[str, str],
        compounding_dicts: dict[str, dict[str, str]]
) -> dict[str, str]:

    processed_texts = dict()

    for title, text in texts.items():

        # Make all replacements specified in compounding dictionary for text
        for current, replacement in compounding_dicts.items():
            text = text.replace(current, replacement)

        # Tokenize Text
        tokenized_text = nltk.to

        # Process pronouns
        pass

    return dict()


def tokenize_texts(
        config: dict,
        texts: dict[str, str]
) -> dict[str, list[str]]:
    pass

    return dict()


def process_texts(
        config: dict,
        tokenized_texts: dict[str, list[str]],
        char_names: dict[str, list[str]]
):
    pass


if __name__ == '__main__':
    main(p.DEFAULT_CONFIG_FILE)
