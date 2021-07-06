# python:alpine is 3.{latest}
FROM python:3.8.5-slim

RUN apt-get -y update && apt-get -y install libgomp1

WORKDIR /src

#pipeline
COPY env.ini .

WORKDIR /src/application

COPY ./application/requirements_freeze.txt .

RUN pip install -r requirements_freeze.txt

COPY ./application . 
#/* ./

EXPOSE 5000

CMD ["python", "app.py"]