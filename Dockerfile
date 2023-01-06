FROM python:3.9

WORKDIR /app

# copy requirements file into docker image
COPY requirements.txt requirements.txt

# install dependencies
RUN pip install -r requirements.txt

# copy all local files to docker image
ADD . .

# run webapp
ENTRYPOINT [ "python3" ]
CMD ["./AMI_Web/app.py"]
