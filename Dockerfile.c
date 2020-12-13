FROM python:3.7.7-slim-buster

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt --ignore-installed

COPY . /app/

CMD ["python3", "/app/control.py", "/app/configs/config.yaml"]
