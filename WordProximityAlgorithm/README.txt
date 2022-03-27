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
