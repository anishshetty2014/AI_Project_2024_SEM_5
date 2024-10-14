import cv2
import numpy
from keras.models import model_from_json

emotion_dict = {0: "Angry", 1:"Disgusted", 2:"Fearfule", 3:"Happy", 4:"Nuetral", 5: "Sad", 6: "Surprised"}

# load json and create_model
json_file = open('emotion_model.json','r')
loaded_model_json=json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("emotion_model.h5")
print("Loaded model from disk")

# start the webcam feed
# cap = cv2.VideoCapture(0)

cap = cv2.Videoapture(0)


