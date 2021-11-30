# set base image (host OS)
FROM teresahgao/superurop-grayscale-cnn:tensorflow-2.3.0

# set the working directory in the container
WORKDIR /

# copy the dependencies file to the working directory
# COPY requirements.txt .

# install dependencies
# RUN pip install -r requirements.txt

# create a mount point where the Docker container can access files on the host system
# VOLUME /

# copy the content of the local directory to the working directory
# COPY / .
# COPY / ./generate_png_from_tiff.py
COPY ./main.py ./main.py
COPY ./Greyscale_Images_png/ ./Greyscale_Images_png/

# RUN mkdir ./Greyscale_Images_png/
# RUN python3 ./generate_png_from_tiff.py
# RUN apt-get update && apt-get install -y vim

# command to run on container start
# CMD [ "python3", "./main.py" ]
