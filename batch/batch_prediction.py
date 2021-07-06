# importing the requests library
import pandas.io.sql as sqlio
import pandas as pd
import configparser
import numpy as np
import requests
import psycopg2
import ast
import boto3

URL = """http://127.0.0.1:5000/prediction_array/"""


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
else:
    pass

if cloud == 1:
    data = redshift("""select * from prediction.data 
                       where class is null or class = '' ;""")
    deldata = redshift("""delete * from prediction.data 
                       where index in """+str(tuple(data['index']))+""";
                       commit;
                       select * from prediction.data 
                       where index in """+str(tuple(data['index'])))                  
                     
else:
    data = pd.read_csv('../ml_model/pictures.csv')

index = list(data['index'])

del data['class'] #Classsification
del data['index'] #Primary Key
del data['proba'] #Found probability

output = pd.DataFrame()
t=10#len(data)
for i in range(t):
    if True:
        #Requests
        datai = np.array(data.iloc[i])
        
        row2store = [index[i]]+list(datai)
        row2store = [str(f) for f in row2store]

        datai = '['+','.join(str(e) for e in datai)+']'
        #print(datai)
        r = requests.get(url = URL+datai)
        
        dic = ast.literal_eval(r.text)
        print(dic['predicted'], dic['probability'])
        print('Predicted: '+r.text)
        row2store = row2store+[dic['predicted']]+[dic['probability']]
      
        output = output.append(pd.DataFrame(row2store).T)
        print('ok')
    else:
        pass   

output.to_csv('output.csv') 
output.to_excel('output.xlsx')

if cloud == 1:
    putOnS3('output.csv', 'output.csv', '/results/')
    copyFromS3('prediction.data', '/results/output.csv')

