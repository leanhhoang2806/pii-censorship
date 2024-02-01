FROM tensorflow/tensorflow:2.3.0-gpu

WORKDIR /app
RUN python -m pip install --upgrade pip
RUN pip install --upgrade pip setuptools

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "train_lm.py"]
