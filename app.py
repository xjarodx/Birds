import os
from flask import Flask, request, jsonify, render_template
#import scrape_wiki
import keras
from keras.preprocessing import image
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
import numpy as np
import pandas as pd

#need to use this in the prepare_image() and upload_file()
img_width, img_height = 150, 150

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

model = None
graph = None


# Loading a keras model with flask
# https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
def load_model(filename):
    test_model = load_model('/jill/birds_model.h5')
    img = load_img(filename,False,target_size=(img_width,img_height))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = test_model.predict_classes(x)
    probs = test_model.predict_proba(x)
    print(preds, probs)
    img_id = test_model.predict_classes(x)[0]

    file = "../data/classes.csv"
    file_df = pd.read_csv(file)

    birdClass = file_df.loc[file_df["id"] == img_id]["name"].unique()[0]

    return birdClass

#load_model(filename)


def prepare_image(img):
    # Convert the image to a numpy array
    img = image.img_to_array(img)
    # Scale from 0 to 255
    img /= 255
    # Invert the pixels
    img = 1 - img
    # Flatten the image to an array of pixels
    image_array = img.flatten().reshape(-1, img_width * img_height)
    # Return the processed feature array
    return image_array

def scrape_wiki(filename):

    bird = load_model(filename)
    all_tables = {}
    bird_data={}
        
    executable_path = {'executable_path': 'chromedriver.exe'}
    browser = Browser('chrome', **executable_path, headless=False)

    my_url = 'https://en.wikipedia.org/wiki/' + bird
    browser.visit(my_url)
    
    ## giving it a bit of time to load before pulling info ###
    time.sleep(1) 
    url_html = browser.html

    ### Table Pull ###

    tables = pd.read_html(url_html)
    #tables[0]
    df = tables[0]
    df.columns = ['About']
    bird_facts_html = df.to_html(index=False, classes="table-hover table-dark table-sm")
    bird_data["facts_table"] = bird_facts_html

    # x = len(tables)
    # x    

    # for i in range (0, x):
    #     table = tables[i]
    #     all_tables[f'tables_{i}']=table

    # scrape_data = tables[0]
    # scrape_data

    ### Image pulls ###

    response = requests.get(url_html)
    soup = bs(response.text, 'html.parser')

    image_tags = soup.findAll('img')

    bird_img = image_tags[1].get("src")
    
    loc_img = image_tags[4].get("src")


@app.route('/post_file', methods=['POST'])
def upload_files():
    if request.method == 'POST':
        print(request)

        if request.files.get('file'):
            # read the file
            file = request.files['file']

            # read the filename
            filename = file.filename

            # create a path to the uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save the file to the uploads folder
            file.save(filepath)

            # Load the saved image using Keras and resize it to the mnist
            # format of 28x28 pixels
            image_size = (img_width, img_height)
            im = image.load_img(filepath, target_size=image_size,
                                grayscale=True)

            # Convert the 2D image to an array of pixel values
            image_array = prepare_image(im)
            print(image_array)

            # Get the tensorflow default graph and use it to make predictions
            #global graph
            #with graph.as_default():

                # Use the model to make a prediction
                #predicted_digit = model.predict_classes(image_array)[0]
                #data["prediction"] = str(predicted_digit)

                # indicate that the request was a success
                #data["success"] = True

                #def scrape_wiki():

            #return jsonify(data)

            return redirect('/')

@app.route('/', methods=['GET'])

def render_page():
    
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

@app.route("/scrape")
def scrape():
    bird_data = scrape_wiki(filename)

    print(bird_data)

    
    coll.update({},bird_data, upsert = True)
    return redirect('/')


if __name__ == "__main__":
    app.run(debug=True)
