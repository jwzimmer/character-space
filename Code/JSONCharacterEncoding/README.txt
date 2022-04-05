JSONCharacterEncoding README

This code looks for capitalized words that aren't following punctuation, then it looks for Noun Phrases that contain an optional prefix and a word that isn't in the dictionary.
It considers these the possible names for the characters.
It takes as input a directory of txt files (books) and outputs json files for each txt file.
The format of the json file is a dict with two dicts in it, the first one contains a list of character names,
the second one has character name: character name pairs.
