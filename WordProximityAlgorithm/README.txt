WordProximityAlgorithm README File

Denis Hudon, denis.hudon@uvm.edu
Julia Zimmerman, julia.zimmerman@uvm.edu

This package of code takes text files and character names, and then generates
data on the distribution of the distance of words from character names in the
text. Words can be filtered based on parts of speech, the intent being to
determine properties of characters by observing their proximity to different
adjectives.

--------------------------------------------------------------------------------

Algorithm Configuration File

The algorithm expects a json format configuration file to be passed to it. If
none is passed, it will use default_config.json. The config file specifies the
following values:

"Input Directory" - The path to the directory containing files to be processed.
If null, default directories specified in parameters will be used.

"Output Directory" - The path to the directory where the results should be
stored. If null, default directories specified in parameters will be used.

"Output File Name" - The name of the results file. This will be a json file -
the .json suffix should not be specified here, it will be added automatically,
along with the date and time the data was generated.

"Remove Stop Words" - Whether or not to remove stop words from the text. See the
NLTK documentation for a list of English stop words.

"Process Pronouns" - Not implemented at this time. Intended to replace pronouns
with the most likely character name. Note that removal of stop words also
removes pronouns...

"Proximity Window" - How far in either direction from a character name to look
for words to collect distance data on. If this value is 10, then the algorithm,
on finding a character name, will look at the 10 previous words and the 10
following words.

"Included Parts of Speech" - A list of NLTK POS tags. Words with these tags will
be used to construct the space into which to embed the character names. Other
words will be ignored.

"Words to Exclude" - A list of words to specifically ignore when processing
texts. Even when these words match the NLTK POS tags, they will not be included
in the results.

--------------------------------------------------------------------------------

Input Directory Structure

The input directory should contain text files containing the text of the books
to embed and ending in '.txt'. Every text file should also have an associated
json file with the same name except for the suffix. This JSON file will contain
the character names to be embedded, as well as a map of transformations to be
performed prior to tokenizing text:

{
  "char_names": [
    "bob",
    "tacoman"
  ],
  "compounding_dict": {
    "robert": "bob",
    "supertaco": "tacoman",
    "the living taco": "tacoman"
  }
}

Here, we see that 'robert' will be replaced with 'bob' in the text prior to
tokenizing, and that 'supertaco' and 'the living taco' will be replaced with
'tacoman'. Then, after tokenizing the text, the embedding values for 'bob' and
'tacoman' will be calculated.

This allows for a character identified with multiple different strings in the
text to be embedded as a single character.

--------------------------------------------------------------------------------

Output Format

The output of the algorithm is a dictionary whose keys are character names, and
whose values are dictionaries. Each of these dictionaries, in turn, has a key
for every word that was matched in all texts with the "Included Parts of Speech"
and that was not include in "Words to Exclude".

The values of these sub-dictionaries are lists of distances at which these words
were found from the character. Hopefully, here, we would expect to see
"Gandalf" have numerous occurrences near "good" and "Sauron" to have numerous
occurrences near "evil", if this process actually reveals underlying structure.

--------------------------------------------------------------------------------
