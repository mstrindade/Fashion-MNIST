# importing tensorflow and keras

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import models, layers

import smtplib, ssl
import email, smtplib, ssl

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import configparser
import boto3

########################### Functions for cloud and warning ###############################

config_ini = configparser.ConfigParser(interpolation=None)
config_ini.read("../env.ini")
cloud = config_ini['environment']['cloud_bin']

if cloud == 1:
    def putOnS3(file, name, path):
        try:
            s3 = boto3.resource('s3',
                aws_access_key_id = config_ini['environment']['aws_access_key_id'],
                aws_secret_access_key = config_ini['environment']['aws_secret_access_key'])
            bucket = s3.Bucket('bucket_name')
            s3.Object('bucket_name', path+'/'+name).put(Body = open(file, 'rb'))
            return "Succsess on save s3!"
        except:
            return "Error on save s3!"

def sendEmail(email, model, acc):
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "ml_team@gmail.com"  
    receiver_email = email
    password = "my_key_4_mail"

    subject = "Warning for training!"
    body = """<html>\
           <body ><p>Dear, </p><p>
            We have found the following problems:<br>

                -Model: <strong style="color: red;">{0}</strong><br>
            -Accuracy: <strong style="color: black;">{1}</strong><br>
            
             </p><p>
             </p>Regards,<br>
             <strong style="color: black;">Automated ML Team</strong><p>
             </body></html>"""
    
    body = body.format(model, acc)

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    # message["Bcc"] =   # Recommended for mass emails

    # Add body to email
    message.attach(MIMEText(body, "html"))
    text = message.as_string()

    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, text)

    sucess = "Email was sent"
    print(body)
    return sucess
#########################################################################################

# Loading the dataset
fashion_mnist=keras.datasets.fashion_mnist # Loading the dataset
(xtrain,ytrain),(xtest,ytest)=fashion_mnist.load_data()

# see the size of the dataset
# print("Train Images Shape: %s \nTrain Labels: %s \nTest Images Shape: %s \nTest Labels: %s"  % (xtrain.shape, xtrain,xtest.shape,ytest))

# Defining array. Each item of array represent integer value of labels. 10 clothing item for 10 integer label.
class_names =['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle-boot']

xtrain = xtrain/255 # So, we are scale the value between 0 to 1 before by deviding each value by 255
xtest = xtest/255 # So, we are scale the value between 0 to 1 before by deviding each value by 255

# One hot encoding of the labels.
#(generally we do one hot encoding of the features in EDA but in this case we are doing it for labels)
ytrain=to_categorical(ytrain)
ytest=to_categorical(ytest)

# Modelling - Model on CNN
# create a sequential model i.e. empty neural network which has no layers in it.
model=models.Sequential()

#==================== Feature Detection / extraction Block ====================#

# Add first convolutional block - To deal with images we use Conv2D and for colour images and shape use Conv3D
#model.add(layers.Conv2D(filters=6, kernal_size(3,3), input_shape=(28,28,1), activation='relu'))
# in the first block we need to mention input_shape
model.add(layers.Conv2D(6,(3,3),input_shape=(28,28,1),activation='relu'))
# Add the max pooling layer
model.add(layers.MaxPooling2D(pool_size=(2,2)))

# Add Second convolutional block
#model.add(layers.Conv2D(filters=6, kernal_size(3,3), activation='relu'))
model.add(layers.Conv2D(10,(3,3),activation='relu'))
# Add the max pooling layer
model.add(layers.MaxPooling2D(pool_size=(2,2)))

#==================== Transition Block (from feature detection to classification) ====================#

# Add Flatten layer. Flatten simply converts matrics to array
model.add(layers.Flatten(input_shape=(28,28))) # this will flatten the image and after this Classification happens

#==================== Classification Block ====================#

# Classification segment - fully connected network
# The Dence layer does classification and is deep neural network. Dense layer always accept the array.
model.add(layers.Dense(128, activation='relu')) # as C5 layer in above image. 
# this 120 is hyper parameter whcih is number of neuron 
#model.add(layers.Dense(84, activation='relu'))# as F6 layer in aboave image

# Add the output layer
model.add(layers.Dense(10, activation='softmax')) # as Output layer in above image. The output layer normally have softmax activation

# if we use softmax activation in output layer then best fit optimizer is categorical_crossentropy
# for sigmoid activation in output layer then loss will be binary_crossentropy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
# if we do not go for One Hot Encoding then use loss='sparse_categorical_crossentropy'

# Train the model 
# Using GPU really speeds up this code
xtrain2=xtrain.reshape(60000,28,28,1)
xtest2=xtest.reshape(10000,28,28,1)

model.fit(xtrain2,ytrain,epochs=5,batch_size=1000,verbose=True,validation_data=(xtest2,ytest))

# evaluate accuracy of the model

test_loss, test_acc = model.evaluate(xtest2, ytest)
print("accuracy:", test_acc)

############################ Save the model in s3 Bucket ########################
if test_acc >= .8:
    model.save('my_model.h5')
    if cloud == 1:
        putOnS3('my_model.h5', 'model.h5', '/models/')
else:
    sendEmail('people@gmail.com', 'fashion_mnist', str(test_acc))        
#################################################################################