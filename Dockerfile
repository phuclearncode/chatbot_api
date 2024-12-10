FROM python:3.12.7

COPY requirements.txt .

RUN pip install -r requirements.txt 

COPY . . 

CMD python app.py 

