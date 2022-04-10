import json
from matplotlib import pyplot as plt

if __name__ == "__main__":
    with open("/Users/jzimmer1/Desktop/character-space/Code/JSONCharacterEncoding/output/prideandprejudice_raw_names.json") as f:
        character_names = json.loads(f.read())
        sorted_char_names = list(character_names.values())
        sorted_char_names.sort(reverse=True)
        #print(len(sorted_char_names))
        #print(len(character_names.values()))
        fig = plt.figure()
        ax = plt.gca()
        ax.scatter(x=range(len(character_names.values())),y=sorted_char_names)
        ax.set_yscale('log')
        ax.set_xscale('log')
        #plt.hist(sorted_char_names)
        plt.show()
    #print(character_names)