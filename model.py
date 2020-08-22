import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import logging
import argparse
import pandas as pd

path = r'C:\Users\Shreyas S K\Desktop\Literature-Search-Engine\metadata.csv'
df = pd.read_csv(path)

df_new = df[['Authors', 'Journal', 'Publish_time', 'Title', 'Abstract', 'URL']]

df_new.dropna(axis = 0, inplace=True)

METHODS = {
    
    'DISTILBERT': {
        'class': "Distilbert",
        'file': None
    }
}

def embedding_class(method: str):
    
    classname = METHODS[method]['class']
    class_ = globals()[classname]
    return class_()

class Distilbert():
    
    def __init__(self):
        self.model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    def cos_sim(self, a, b):

        simi = np.dot(a,b) / np.multiply(np.linalg.norm(a),np.linalg.norm(b))
        return simi

    def distil_embeddings(self):
        vector1 = torch.load('distil(1-60k).pt')
        vec = np.array(vector1)
        a = torch.from_numpy(vec)
        vector2 = torch.load('distil(60k-1.12l).pt')
        b = torch.from_numpy(vector2)
        c = torch.cat((a,b), 0)
        return a

    def top_similar_sentences(self, df, sentence):

        
        sent = self.model.encode(sentence)
        embed = self.distil_embeddings()
        sim = []
        for j, batch in enumerate(embed):
            similarity = self.cos_sim(sent, batch)
            sim.append(similarity)
        op = sorted(sim, reverse=True)
        index = [sim.index(f) for f in op[:5]]
        dataframe = df.iloc[index]

        return dataframe
    
def output(method: str, df, text):
    
    model = embedding_class(method)
    op = model.top_similar_sentences(df, text)
    logging.info("Model Prediction Successful")
    
    return op

def main(samples):
    
    # Get list of available methods:
    method_list = [method for method in METHODS.keys()]
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', type=str, help="Enter one or more methods \
                        (Choose from following: {})".format(", ".join(method_list)),
                        required=True)
    args = parser.parse_args()

    for method in args.method:
        if method not in METHODS.keys():
            parser.error("Please choose from the below existing methods! \n{}".format(", ".join(method_list)))
            
        print("Method: {}".format(method.upper()))
        
        text1 = samples[0]
        sim = output(method, df_new, text1)
            
        #print("Top 5 Papers are - ", sim)  

if __name__ == "__main__":
    # Evaluation text
    samples = "The cast is uniformly excellent ... but the film itself is merely mildly charming ."
    main(samples)


















