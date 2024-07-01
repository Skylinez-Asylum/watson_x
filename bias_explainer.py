from nltk import sent_tokenize
from ibm_cloud_sdk_core import IAMTokenManager
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator,BearerTokenAuthenticator
import os, getpass
import requests

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

class BiasExplainer:

    def __init__(self):
        pass

    def bias_explain(self,input_text,input_bias,summary):
        model_id="ibm/granite-13b-instruct-v2"
        parameters = {
            "decoding_method": "greedy",
            "max_new_tokens": 150,
            "min_new_tokens": 50,
            "repetition_penalty": 2
        }
        access_token = IAMTokenManager(
             apikey = 'ebcUPL7y5aKMOCd4kzazKv2ghfxAtLXggRKAdklMp9fu',
             url = "https://iam.cloud.ibm.com/identity/token"
             ).get_token()
        project_id = "166da5bd-8f3f-4048-99bb-346034ffc96c"
        prompt = Prompt(access_token, project_id)
        from collections import defaultdict
        bias_categories = defaultdict(list)
        explainer_instruction = """You are agent that explains in 50 words how the biased statement and corresponding Bias Type affects the summary provided in less than 50 words :
         """
        full_detection_input = f"{explainer_instruction}\n\nSummary:{summary}Bias Type:{input_bias}\nBiased Statement:{input_text}"
        detected_bias = prompt.generate(full_detection_input, model_id, parameters).replace("\n", " ")
        return detected_bias

    def run(self,input_text,input_bias,summary):
        return self.bias_explain(input_text,input_bias,summary)   