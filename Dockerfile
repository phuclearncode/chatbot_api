FROM python:3.12.7-slim

WORKDIR /chatbot

RUN  pip install --no-cache-dir -r /chatbot/requirements.txt 

COPY /chatbot/app.py .

EXPOSE 7860

CMD ["python", "app.py"]

