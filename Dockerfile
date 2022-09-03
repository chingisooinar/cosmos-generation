# Install some basic utilities
FROM python:3.8
RUN apt-get update  
RUN apt-get install ffmpeg libsm6 libxext6  -y


# 
COPY ./requirements.txt /app/requirements.txt

# 
#RUN apt-get update
#RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
# 
COPY ./app /app
COPY ./model.pkl /app/model.pkl

#
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

