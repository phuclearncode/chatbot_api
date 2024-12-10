FROM python:3.12.7-slim

WORKDIR /chatbot_api

RUN  pip install --no-cache-dir -r /chatbot_api/requirements.txt 

COPY /chatbot_api/app.py .

EXPOSE 7860

CMD ["python", "app.py"]

