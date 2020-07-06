from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from flask_restful import Resource,Api,reqparse
from flask_cors import CORS
from flask import *
# import tensorflow as tf
# import pathlib
# import matplotlib.pyplot as plt
# import pandas as pd
app = Flask(__name__)
api = Api(app)

app.config['SECRET_KEY'] = 'mykey'

########################
###db section###########
########################

cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
@app.route('/')
def index():
    return "<center><h1>Hello This is Mgahed</h1></center>"

class test(Resource):
    def post(self):
        label = "Not assigned"
        face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        classifier =load_model('Emotion_little_vgg.h5')

        class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

        # def face_detector(img):
        #     # Convert image to grayscale
        #     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #     faces = face_classifier.detectMultiScale(gray,1.3,5)
        #     if faces is ():
        #         return (0,0,0,0),np.zeros((48,48),np.uint8),img

        #     for (x,y,w,h) in faces:
        #         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #         roi_gray = gray[y:y+h,x:x+w]

        #     try:
        #         roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        #     except:
        #         return (x,w,y,h),np.zeros((48,48),np.uint8),img
        #     return (x,w,y,h),roi_gray,img


        cap = cv2.VideoCapture(0)



        while True:
            # Grab a single frame of video
            ret, frame = cap.read()
            labels = []
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
            # rect,face,image = face_detector(frame)


                if np.sum([roi_gray])!=0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi,axis=0)

                # make a prediction on the ROI, then lookup the class

                    preds = classifier.predict(roi)[0]
                    label=class_labels[preds.argmax()]
                    label_position = (x,y)
                    cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                else:
                    cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            cv2.imshow('Emotion Detector',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        somedict = {
                        "label" : label
                   }
        return somedict
api.add_resource(test, '/api/test')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__=='__main__':
    app.run(debug=True)
