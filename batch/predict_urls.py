# importing the requests library
import requests

image = "https://static.zattini.com.br/produtos/sapato-social-derby-polo-state-tradicionally-masculino/06/ITO-2329-006/ITO-2329-006_zoom1.jpg"
image = image.replace("/", "8abc8")


#EndPoint URL
URL = """http://127.0.0.1:5000/prediction_url/"""

#Requests
r = requests.get(url = URL+image)
print(r.text)
 