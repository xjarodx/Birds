import os
from flask import Flask, request, jsonify, render_template
import scrape_wiki

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
def load_model():
    test_model = load_model('/jill/birds_model.h5')
    img = load_img('image_to_predict.jpg',False,target_size=(img_width,img_height))
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

#load_model()


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



@app.route('/', methods=['GET', 'POST'])

def upload_file():
    data = {"success": False}
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
            

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
