from nltk import sent_tokenize
from ibm_cloud_sdk_core import IAMTokenManager
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator,BearerTokenAuthenticator
import os, getpass
import requests
#import sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#from backend.prompt import Prompt

class Prompt:
    def __init__(self, access_token, project_id):
        self.access_token = access_token
        self.project_id = project_id

    def generate(self, input, model_id, parameters):
        wml_url = "https://us-south.ml.cloud.ibm.com/ml/v1-beta/generation/text?version=2023-05-28"
        Headers = {
            "Authorization": "Bearer " + self.access_token,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        data = {
            "model_id": model_id,
            "input": input,
            "parameters": parameters,
            "project_id": self.project_id
        }
        response = requests.post(wml_url, json=data, headers=Headers)
        if response.status_code == 200:
            return response.json()["results"][0]["generated_text"]
        else:
            return response.text


class BiasFinder:

    def __init__(self):
        pass   
   
    def detect_bias(self,input_text):
        uniq=set(['Confirmation Bias',"Selection Bias","Cognitive Bias","Publication Bias"])
        model_id="ibm/granite-13b-instruct-v2"
        parameters = {
            "decoding_method": "greedy",
            "max_new_tokens": 1000,
            "min_new_tokens": 1,
            "repetition_penalty": 1
        }
        access_token = IAMTokenManager(
             apikey = 'ebcUPL7y5aKMOCd4kzazKv2ghfxAtLXggRKAdklMp9fu',
             url = "https://iam.cloud.ibm.com/identity/token"
             ).get_token()
        project_id = "166da5bd-8f3f-4048-99bb-346034ffc96c"
        prompt = Prompt(access_token, project_id)
        from collections import defaultdict
        bias_categories = defaultdict(list)
        detection_instruction = """Your task is to identify bias and classify it as one of the following:
            
            - Confirmation Bias
            - Selection Bias
            
            - Publication Bias
            - Cognitive Bias
            
            output the category of the bias. If no bias is found, return "no bias"."""
        
        sentences = sent_tokenize(input_text)
        for sentence in sentences:
            full_detection_input = f"{detection_instruction}\n\n{sentence}"
            detected_bias = prompt.generate(full_detection_input, model_id, parameters).replace("\n", "")
            if detected_bias in uniq:
                if len(sentence)>=5:
                    bias_categories[detected_bias].append(sentence)
        print(bias_categories)
        return bias_categories

    def run(self,input_text):
        return self.detect_bias(input_text)    

if __name__ == "__main__":
    bias = BiasFinder() 
    text = """Men are naturally better leaders than women because they are more rational and less emotional.
             Asian students often struggle with creative tasks compared to their Western counterparts.
              People from wealthy backgrounds are inherently more intelligent and capable of success. 
              Younger employees are always more innovative than older ones. Women should primarily focus on homemaking rather than pursuing careers. 
               African Americans are more prone to criminal behavior than other races. People with disabilities can't contribute effectively in a fast-paced work environment. Christians have higher moral standards than people of other religions. Gay men are typically more stylish and flamboyant than heterosexual men. Republicans have better family values compared to Democrats."""
    s = bias.run(text)  
    print(s)     