from bias_finder import BiasFinder
import nltk
from collections import defaultdict
from bias_explainer import BiasExplainer
nltk.download('punkt')
if __name__ == "__main__":
    bias = BiasFinder()
    explainer = BiasExplainer()
    text = """Men are naturally better leaders than women because they are more rational and less emotional.
             Asian students often struggle with creative tasks compared to their Western counterparts.
              People from wealthy backgrounds are inherently more intelligent and capable of success. 
              Younger employees are always more innovative than older ones. Women should primarily focus on homemaking rather than pursuing careers. 
               African Americans are more prone to criminal behavior than other races. People with disabilities can't contribute effectively in a fast-paced work environment. Christians have higher moral standards than people of other religions. Gay men are typically more stylish and flamboyant than heterosexual men. Republicans have better family values compared to Democrats."""
    bias_data = bias.run(text)
    

    # Loop through each key (bias type) in the defaultdict
    for bias_type, sentences in bias_datas():
        for sentence in sentences:
            print(explainer.run(sentence,bias_type))