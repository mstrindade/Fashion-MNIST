#######################################################################################
########                            fashion mnist                             #########
#######################################################################################

# Importing all necessary packages for cotton-web-app
from sys import float_repr_style
from flask import Flask,   jsonify,   render_template,  request, \
                  flash,   redirect,  session,          abort

import pandas as pd

pd.options.display.max_columns = None
pd.set_option('display.max_colwidth', None)
from IPython.core.display import display, HTML, clear_output        
display(HTML("<style>.container { width:90% !important; }</style>"))

import numpy as np
import pandas.io.sql as sqlio
import psycopg2
import json
import re

import urllib
import cv2
import urllib.request

import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"

from datetime import datetime, timedelta
import os
import folium
from folium import plugins
import random
import pickle

import smtplib, ssl
import email, smtplib, ssl

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import tensorflow as tf

import configparser
import boto3

config_ini = configparser.ConfigParser(interpolation=None)
config_ini.read("../env.ini")
cloud = config_ini['environment']['cloud_bin']

# Connectiong the data basis.
if cloud == 1:
    def redshift(sql):
        conn = psycopg2.connect(dbname = config_ini['environment']['redshift_dbname'],
                                user = config_ini['environment']['redshift_user'],
                                password = config_ini['environment']['redshift_password'],
                                host = config_ini['environment']['redshift_server'],
                                port = config_ini['environment']['redshift_port'])
        data = sqlio.read_sql_query(sql, conn)
        conn.close()
        return data
else:
    def redshift(sql):
        pass

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

def copyFromS3(redshift_table, s3_file):
    try:
        redshift("""COPY """+redshift_table+"""
                    from 's3://bucket_name/'"""+s3_file+"""
                    iam_role """+config_ini['environment']['redshift_hole']+"""
                    ignoreheader 1
                    delimiter ',';
                    commit;
                    select * from """+redshift_table+""" limit 10""")
        return "Succsess on copy Redshift!"
    except:
        return "Error on copy Redshift!"                

pd.options.display.max_columns = None

# ##################################################################################
# Set application
application = Flask(__name__)

#Get port from environment variable or choose 5050 as local default
port = int(os.getenv("PORT", 5050))

application.secret_key = os.urandom(12)
# ##################################################################################

#file_name = "modelo.pkl"
#xgb_model_loaded = pickle.load(open(file_name, "rb"))

####################################################################################
#                      Smart functions for this application                        #
####################################################################################
def str2date(data):
    return datetime.strptime(data, '%Y-%m-%d').date()

def date2str(data):
    return data.strftime("%Y-%m-%d")

def sumdays(data, days):
    if type(data) == str:
        data = str2date(data)
    
    data = (data + timedelta(days=days)).strftime("%Y-%m-%d")
    return data

###############################################################################
########                   End Point GET POST                         #########
###############################################################################

@application.route("/isAlive")
def index():
    return "I am working!"

@application.route('/test/<name>-<age>')
def index2(name,age):
    return 'Ok ' + name + ', your age is ' + age

class_names =  ['T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat',
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle-boot']

def string2list(string):
    string = string.replace('[','').replace(']','')
    li = list(string.split(","))
    final = [float(i) for i in li]
    return final
  
@application.route('/prediction_array/<value>')
def prediction_array(value):
    dic = {}
    
    # Recreate the exact same model, including its weights and the optimizer
    new_model =  tf.keras.models.load_model('../ml_model/my_model.h5')
    value = string2list(value)
    figure = np.array(value).reshape(1,28,28,1)
    p = new_model.predict(figure)
    pred = class_names[np.argmax(p)]
    
    dic['predicted'] = pred
    dic['probability'] = str(max(p[0]))
    
    return dic

@application.route('/prediction_url/<url>')
def prediction_url(url):
    
    dic = {}
    # Recreate the exact same model, including its weights and the optimizer
    new_model =  tf.keras.models.load_model('../ml_model/my_model.h5')
    
    url = url.replace("8abc8", "/")
    url_response = urllib.request.urlopen(url)
    img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, -1)

    dim = (28,28)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    figure = gray.reshape(1,28,28,1)/225
  
    p = new_model.predict(figure)
    pred = class_names[np.argmax(p)]
    
    dic['predicted'] = pred
    dic['probability'] = str(max(p[0]))

    return dic
      
@application.route('/prediction_example/<value>')
def prediction(value):
    model = pickle.load(open('model.pkl', 'rb'))
    pred = model.predict([[float(value)]])
    return str(pred)

@application.route('/prediction/', methods=['GET','POST'])
def get_prediction():
    feature = float(request.args.get('f'))
    feature2 = request.args.get('s')
    model = pickle.load(open('model.pkl', 'rb'))
    pred = model.predict([[feature]])
    return feature2 + str(pred)



####################################################################################

@application.route('/')
def home():
    #If we want do develop a login page
    #return render_template('loginapp.html')
    return page1()

####################################################################################
####                         Codes for page page1.html                          ####
####################################################################################
@application.route("/page1")
def page1():
    data = plot1('var11','var2','var3')
    return render_template('page1.html', data = data)

@application.route('/caminhos_variables', methods=['GET', 'POST'])
def caminhos_variables(mac = 14, order = 'seq', variable='max'):
    mac = request.args['sub']
    order = request.args['ord']
    variable = request.args['variable']
    graphJSON = plot1(mac,order,variable)
    return graphJSON

def plot1(var1, var2, var3):

    inicio = '2021-01-01'
    dates = []
    for i in range(365):
        dates.append(sumdays(inicio,i))

    #dates = ['2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01',
    #     '2021-08-01', '2021-09-01', '2021-10-01', '2021-11-01', '2021-12-01']


    today = date2str(datetime.now())
    dd = '2021-08-01'
    ff = '2021-09-02'
    hh = '2021-10-01'
    
    x = np.linspace(-5, 5, num=365)

    today_index = dates.index(today)

    mean = (1+np.cos(-x[0:today_index]))*4500+100
    stdA = (1+np.sin(-x[0:today_index]))*4500+200
    stdB = (1+np.sin(-x[0:today_index]))*4500+1
    
    Pmean = 1/(1+np.cos(-x[today_index-1::]))*4500+100
    PstdA = 1/(1+np.sin(-x[today_index-1::]))*4500+200
    PstdB = 1/(1+np.sin(-x[today_index-1::]))*4500+1

    fig = go.Figure()
    #fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Create and style traces
    fig.add_trace(go.Scatter(x=dates[0:today_index], y=stdA, name='ref_factor',
                            line=dict(color='blue', width=2), showlegend=False))
    fig.add_trace(go.Scatter(x=dates[0:today_index], y=mean, name = 'Real',
                            line=dict(color='red', width=4)))
    fig.add_trace(go.Scatter(x=dates[0:today_index], y=stdB, name='ref_factor',
                            line=dict(color='blue', width=2), showlegend=False))
    fig.add_trace(go.Scatter(x=dates[today_index-1::], y=PstdA, name='ref_factor',
                            line=dict(color='blue', width=2,dash='dot'), showlegend=False))
    fig.add_trace(go.Scatter(x=dates[today_index-1::], y=Pmean, name = 'Predicted',
                            line=dict(color='red', width=4,dash='dot')))
    fig.add_trace(go.Scatter(x=dates[today_index-1::], y=PstdB, name='ref_factor',
                            line=dict(color='blue', width=2,dash='dot'), showlegend=False))

    #temp = [12.7, 14.3, 18.6, 35.5, 49.9, 58.0, 60.0, 58.6, 51.7, 45.2, 32.2, 29.1]
    #when = [sumdays('2021-01-01',i*30) for i in range(12)]
    #fig.add_trace(go.Scatter(x=when, y=temp, name='Temperature', line=dict(color='green', width=2, dash='dot')), secondary_y=True)

    window = 7

    # Edit the layout
    fig.update_layout(
                    yaxis_title='Number of calls')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', 
                    xaxis = dict(
                    tickmode = 'array',
                    tickvals = [today, dd, ff, hh],
                    ticktext = ['today<br>'+today, 'dayHUINY<br>'+dd, 'dayOIUJ<br>'+ff, 'dayKHVB<br>'+hh]),
                    hovermode='x',
                    legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.09,
                    xanchor="right",
                    x=1
                    ), 
                    margin=dict(l=5, r=5)
                    )
    graphJSON3 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON3

@application.route('/caminhos_map', methods=['GET', 'POST'])
def caminhos_map(mac = 14, order = 'seq', variable='max'):
    mac = int(request.args['sub'])
    order = request.args['ord']
    variable = request.args['variable']
    graphJSON = plot2(mac,order,variable)
    return graphJSON

def plot2(v1,v2,v3):
    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_us_cities.csv')
    df.head()

    df['text'] = df['name'] + '<br>Access ' + (df['pop']/1e6).astype(str)+' million'
    limits = [(0,2),(3,10),(11,20),(21,50),(50,3000)]
    colors = ["royalblue","crimson","lightseagreen","orange","lightgrey"]
    cities = []
    scale = 5000

    fig = go.Figure()

    for i in range(len(limits)):
        lim = limits[i]
        df_sub = df[lim[0]:lim[1]]
        fig.add_trace(go.Scattergeo(
            locationmode = 'USA-states',
            lon = df_sub['lon'],
            lat = df_sub['lat'],
            text = df_sub['text'],
            marker = dict(
                size = df_sub['pop']/scale,
                color = colors[i],
                line_color='rgb(40,40,40)',
                line_width=0.5,
                sizemode = 'area'
            ),
            name = '{0} - {1}'.format(lim[0],lim[1])))

    fig.update_layout(
            title_text = 'US city accessing <br>(Click legend to toggle traces)',
            showlegend = True,
            geo = dict(
                scope = 'usa',
                landcolor = 'rgb(217, 217, 217)',
            )
        )
    
    graphJSON3 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON3


#############################################################################

if __name__ == "__main__":
    application.run(host = '127.0.0.1', debug=True)