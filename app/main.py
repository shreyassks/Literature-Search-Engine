# -*- coding: utf-8 -*-

from app import app
from flask import render_template, request
from model import output
import pandas as pd

path = '/Users/shreyassk/Covid-app/metadata.csv'
df = pd.read_csv(path)

df_new = df[['source_x','title','abstract','publish_time','authors','journal','url']]

df_new.dropna(axis = 0, inplace=True)

print(df_new.shape)


@app.route('/')
def main():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def index():
    exp = ""
    if request.method == 'POST':
        text_1 = request.form['entry_1']
        method = request.form['model']
        if any(not v for v in [text_1]):
            raise ValueError("Please do not leave text fields blank.")
        
        if method != "base":
            exp = output(method, df_new, text_1)
         
            
    return render_template('index.html', exp=exp, entry_1=text_1, embed=method)