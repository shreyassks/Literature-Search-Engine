# -*- coding: utf-8 -*-

from app import app
from flask import render_template, request
from model import output
import pandas as pd

path = 'metadata.csv'
df = pd.read_csv(path)

df_new = df[['Authors', 'Journal', 'Publish_time', 'Title', 'Abstract', 'URL']]

df_new.dropna(axis = 0, inplace=True)

pd.set_option('colheader_justify', 'center')
pd.set_option('display.max_colwidth', -1)

def make_clickable(val):
    # target _blank to open new window
    return '<a target="_blank" href="{}">{}</a>'.format(val, val)

df_new.style.format({'url': make_clickable})

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
            exp.style.format({'url': make_clickable})
            exp.style.set_properties(**{'background-color': 'black',
                           'color': 'lawngreen',
                           'border-color': 'white'})
            
    return render_template('index.html', exp = exp, entry_1=text_1, embed=method)
