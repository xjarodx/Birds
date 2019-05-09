import os
from flask import Flask, request, jsonify, render_template, redirect, send_from_directory
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime as dt
import requests
import time
from splinter import Browser
from bs4 import BeautifulSoup as bs

import keras
from keras.preprocessing import image
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route('/', methods=['GET', 'POST'])

def render_page():
    print("made it to render page")
    obj1 = {
        "bird": {
            "facts_table": [
                'fact 1',
                'fact 2',
                'fact 3',
                'fact 4'
            ],
            "safe_table": [
                'fact 1',
                'fact 2',
                'fact 3',
                'fact 4'
            ],
            "img": "http://www.google.images/image.jpg"
        }
    }
    
    return render_template('index.html', bird_data=obj1)


@app.route('/postfile', methods=['POST'])

def upload_files():
    
    global filepath

    if request.method == 'POST':
        print(request)
        print('made it to postfile and req meth does equal post')
        if request.files.get('file'):
            # read the file
            file = request.files['file']

            # read the filename
            filename = file.filename

            # create a path to the uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save the file to the uploads folder
            file.save(filepath)
           
            #g=filepath

            time.sleep(1)
            #print(g)

            return redirect('/processImg')

    else:
        return render_template('index.html')



@app.route('/processImg', methods=['GET','POST'])

def predict():
    
    print ("made it to load model")
    
    model = load_model('jill/birds_model5.h5')
    graph = None

    img_width, img_height = 150, 150

    img = load_img(filepath,False,target_size=(img_width,img_height))

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    preds = model.predict_classes(x)
    probs = model.predict_proba(x)
    print(preds, probs)

    img_id = model.predict_classes(x)[0]

    classFile = "data/classes.csv"
    file_df = pd.read_csv(classFile)

    birdClass = file_df.loc[file_df["id"] == img_id]["name"].unique()[0]
    print (birdClass)

    print("made it to the scrape")
    all_tables = {}
    bird_data={}
        
    executable_path = {'executable_path': 'chromedriver.exe'}
    browser = Browser('chrome', **executable_path, headless=False)

    my_url = 'https://en.wikipedia.org/wiki/' + birdClass
    browser.visit(my_url)
    print(my_url)   
    
    url_html = browser.html

    ### Table Pull ###

    tables = pd.read_html(url_html)

    df = tables[0]
    df.rename(index=str, columns={0:"Ab", 1: "info"}, inplace=True)
    df1=df.dropna(subset=["Ab"])
    df2=df1.replace(np.nan, '', regex=True)
    df2["About"]=df2["Ab"].map(str) + df2["info"].map(str)
    df3=df2.drop(columns=['Ab', 'info'])

    print(df3)
    
    #df1=df['About'].combine_first(df['info'])
    #print(df1)

    bird_facts_html = df3.to_html(index=False, classes="table-light table-hover table-sm", justify='center')
    bird_data["facts_table"] = bird_facts_html
    print('Got the data')
    
    ### Image pulls ###

    response = requests.get(my_url)
    soup = bs(response.text, 'html.parser')

    image_tags = soup.findAll('img')

    bird_img = image_tags[0].get("src")
    print(bird_img)
    loc_img = image_tags[3].get("src")
    print(loc_img)
    browser.quit()
    return render_template('return.html', bird_data=bird_data, bird_img=bird_img, bird_loc=loc_img, birdClass=birdClass)

if __name__ == "__main__":
    app.run(debug=True)
