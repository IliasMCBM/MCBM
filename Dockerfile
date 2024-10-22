FROM ubuntu:22.04

WORKDIR /MCBM

RUN apt-get update && apt-get install -y python3 python3-pip

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

RUN pip3 install -r requirements.txt

CMD [ "streamlit", "run", "ocr.py" ]


