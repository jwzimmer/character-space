import re
import pandas as pd
import json
#pip install scikit-learn
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

prefixes = ["mr.", "mrs.", "ms.", "dr.", "mme.", "rev.", "lt.", "col.",
            "hon.", "st."]

def pad_prefixes(some_text, prefix_list):
    for prefix in prefixes:
        some_text = re.sub(re.escape(prefix),prefix[:-1],some_text)
    return some_text

def run_LSA(text, character_names, prefix_list):
    # following steps in https://towardsdatascience.com/latent-semantic-analysis-deduce-the-hidden-topic-from-the-document-f360e8c0614b
    text = text.lower()
    text = pad_prefixes(text, prefix_list)
    documents = text.split(".|!|?")
    df = pd.DataFrame()
    df["documents"] = documents
    # remove special characters
    df['clean_documents'] = df['documents'].str.replace("[^a-zA-Z#]", " ")
    # tokenization
    tokenized_doc = df['clean_documents'].fillna('').apply(lambda x: x.split())
    # remove stop-words
    stop_words = stopwords.words('english')
    stop_words += [x[:-1] for x in prefix_list]
    tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
    # de-tokenization
    detokenized_doc = []
    for i in range(len(df)):
        t = ' '.join(tokenized_doc[i])
        detokenized_doc.append(t)
    df['clean_documents'] = detokenized_doc

    # make a df for each character -- if their name is in a document, put that document in their df
    character_dict_list = []
    for character in character_names:
        new_dict = {'clean_documents':[]}
        for row in df.iterrows():
            contents = row[1]['clean_documents']
            pattern1 = re.compile(r'\b'+character+r'\b')
            pattern2 = re.compile(r'\b'+character+r'\'*s+\b')
            if (pattern1.search(contents) or pattern2.search(contents)):
                new_dict['clean_documents'].append(contents)
        if len(new_dict['clean_documents']) == 0:
            character_dict_list += ["No contents"]
        else:
            character_dict_list.append(new_dict)
    character_df_list = [pd.DataFrame().from_dict(x) for x in character_dict_list if x != "No contents"]

    lsa_output_list = []
    topic_encoded_df_list = []
    encoding_matrix_list = []
    for i in range(len(character_df_list)):
        char_df = character_df_list[i]
        char_name = character_names[i]
        if type(char_df) != "str":
            #vectorizer = TfidfVectorizer(stop_words='english', smooth_idf=True)
            vectorizer = TfidfVectorizer(smooth_idf=True)
            X = vectorizer.fit_transform(char_df['clean_documents'])
            # we want 1 LSA/ SVD topic per set of documents, and 1 set of documents per character name
            svd_model = TruncatedSVD(n_components=1, algorithm='randomized', n_iter=100, random_state=122)
            lsa = svd_model.fit_transform(X)
            lsa_output_list.append(lsa)
            topic_encoded_df = pd.DataFrame(lsa, columns=[char_name])
            topic_encoded_df["documents"] = df['clean_documents']
            topic_encoded_df_list.append(topic_encoded_df)
            dictionary = vectorizer.get_feature_names_out()
            encoding_matrix = pd.DataFrame(svd_model.components_, index=[char_name], columns=(dictionary)).T
            encoding_matrix = encoding_matrix.sort_values(char_name,axis=0)
            encoding_matrix_list.append(encoding_matrix.T)
        else:
            lsa_output_list += ["No contents"]
            topic_encoded_df_list += ["No contents"]
            encoding_matrix_list += ["No contents"]
    #print(pd.concat([encoding_matrix_list[0],encoding_matrix_list[1]]))
    return encoding_matrix_list

def use_LSA_words_in_matrix(encoding_matrix_list, character_names):
    new_df = pd.DataFrame()
    for df in encoding_matrix_list:
        new_df = pd.concat([new_df,df])
    new_df = new_df.fillna(0)
    #print(new_df.shape, new_df)
    return new_df

if __name__ == '__main__':
    #dirname = "/home/denis/PycharmProjects/character-space/WordProximityAlgorithm/Example/example_input"
    filename = "/Users/jzimmer1/Documents/GitHub/character-space/TestDirectory/prideandprejudice.txt"
    text = open(filename)
    text = text.read()

    character_names = None
    with open("/Users/jzimmer1/Documents/GitHub/character-space/prideandprejudice.json") as f:
        character_names = json.loads(f.read())
    #print(character_names)
    character_names = character_names["char_names"]
    enc_matrix_list = run_LSA(text, character_names, prefixes)
    output_df = use_LSA_words_in_matrix(enc_matrix_list, character_names)
    output_df.to_json("lsa.json")
