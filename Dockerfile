FROM python:3.10-slim

MAINTAINER Isakov Maxim Isakovmv.ekb@gmail.com

WORKDIR /app

ENV TZ=Asia/Yekaterinburg
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY . .

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 unzip wget -y

RUN wget -O ./models/OcrNet.pth "https://www.dropbox.com/s/76qbi0smu7v836t/OcrNet.pth"
RUN wget -O ./models/yolov5m.pt "https://www.dropbox.com/s/j53f8athnygqu6s/yolov5m.pt"

RUN unzip -o ./models/yolov5.zip -d ./models/yolov5

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "main.py"]
