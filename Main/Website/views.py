from django.shortcuts import render
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import os
# Create your views here.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
h5_file_path = os.path.join(BASE_DIR,'cnn.h5')
model=load_model(h5_file_path)
def home(request):
    
    return render(request,"home.html")

def predict(request):
    if request.method == 'POST':
        # Get the uploaded image file from request.FILES
        image_file = request.FILES.get("file")

        # Open the image using Pillow (PIL)
        img = Image.open(image_file)

        # Resize the image to fit the model input size (e.g., 28x28)
        img = img.resize((28, 28))

        # Convert the image to a numpy array
        img = np.array(img)

        # Normalize the image
        img = img / 255.0  

        # Make prediction
        # Assuming you have loaded your CNN model and assigned it to `model`
        prediction = model.predict(np.expand_dims(img, axis=0))
        predicted_class = np.argmax(prediction)

        # Replace these labels based on your classes
        class_labels = {0: 'Class_0', 1: 'Class_1'}  

        result = {"class": class_labels[predicted_class]}
        return render(request, "home.html", {'acc': result})

    return render(request, "home.html")