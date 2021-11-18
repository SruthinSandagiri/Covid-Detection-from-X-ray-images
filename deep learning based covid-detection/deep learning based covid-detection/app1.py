# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 17:00:26 2021

@author: sandagiri
"""

import io
import string
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import flask
from flask import Flask, jsonify, request, render_template, redirect
#from PIL import Image
#import Image
import PIL.Image #import core as image
import os
from werkzeug.utils import secure_filename

print(flask.__version__)

app = Flask(__name__)

# Modelling Task
model = models.resnet50()
num_inftr = model.fc.in_features
model.fc = nn.Linear(in_features = 2048, out_features = 3)
model.load_state_dict(torch.load("C:\\Users\\sandas\\Documents\\uni-project\\resnet50.pt"))
model.eval()

class_names = class_names = ['normal', 'viral', 'covid']

def transform_image(image_bytes):
	my_transforms = transforms.Compose([
         #transforms.ToPILImage(),
		transforms.Resize(size = (224, 224)),
        transforms.CenterCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

	])

	image_1 = PIL.Image.open(io.BytesIO(image_bytes))
	return my_transforms(image_1).unsqueeze(0)

def get_prediction(image_bytes):
	tensor = transform_image(image_bytes=image_bytes)
	outputs = model.forward(tensor)
	_, prediction = torch.max(outputs, 1)
	return class_names[prediction]

@app.route('/', methods=['GET','POST'])
def index():
    # Main page
    return render_template('index.html')


#@app.route('/predict', methods=['GET', 'POST'])
#def upload():
#    if request.method == 'POST':
#        # Get the file from post request
#        f = request.files['file']

        # Save the file to ./uploads
#        basepath = os.path.dirname(__file__)
#        file_path = os.path.join(
#            basepath, 'uploads', secure_filename(f.filename))
#        f.save(file_path)
#
 #       # Make prediction
#        preds = get_prediction(img_bytes)
 #       result=preds
 #       return result
 #   return None
 
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        file = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(file.filename))
        file.save(file_path)
        print("sruthin")

        # Make prediction
        img_bytes = file.read()
        print("hello")
        prediction_name = get_prediction(img_bytes)
        print("hi")
        return prediction_name
    return None

#@app.route('/', methods=['GET', 'POST'])
#def upload_file():
#	if request.method == 'POST':
#		if 'file' not in request.files:
#			return redirect(request.url)
#		file = request.files.get('file')
#		if not file:
#			return
#		img_bytes = file.read()
#		prediction_name = get_prediction(img_bytes)
#		return render_template('result.html', name=prediction_name.lower())#, description=diseases[prediction_name])

#	return render_template('index.html')


if __name__ == '__main__':
    app.run()
