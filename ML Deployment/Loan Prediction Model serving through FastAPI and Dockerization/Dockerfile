FROM python:3.8

WORKDIR /code

COPY ./requirement.txt /code/requirement.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirement.txt

COPY ./app/model.pkl /code/app/model.pkl

COPY ./app/model_columns.pkl  /code/app/model_columns.pkl 

COPY ./app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
