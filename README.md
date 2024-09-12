# Serving the Tracking Models on the Jetson

The tracking pipeline on the Jetson is implemented using 
[TensorFlow Serving](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple).
The server is designed to run in the background, and the ROS nodes that 
perform tracking make gRPC calls to this server.

The server is managed using `docker compose`. In order to set it up, you will
first need to build the Docker container on the Jetson:

```shell
MODEL_DIR="" docker compose build
```

Note that, as of this time, there is no binary TensorFlow Serving package
available for the Jetson. Therefore, this Docker build process will compile
it from source, which generally takes several hours. I suggest you run it
overnight.

## Converting the Models

Realtime tracking depends on models converted using TensorRT for fast 
inference. This conversion will need to be performed every time you update
the models. These instructions assume that you have uploaded the raw models
to `~/tf_models_temp` on the Jetson. This directory should have the 
following structure:

```shell
- tf_models_temp
 |__ detection_model
 |__ small_detection_model
 |__ tracking_model
```

Each of these three models should be stored in the TensorFlow `saved_model` 
format. See `notebooks/model_to_tf_saved.ipynb`.

Converting the models can be done (mostly) automatically:

```shell
INPUT_DIR=/home/mars/tf_models_temp/ MODEL_DIR=/media/mars/Data/models/trt_models docker compose -f docker-compose-conversion.yml up
```

Here, `INPUT_DIR` is the directory containing the input models, and 
`MODEL_DIR` is the desired output directory. This can be anything you want. 
You just have to point TensorFlow Serving at it later. This conversion process
might take awhile, up to an hour or so.

The conversion script can automatically recognize whether previous versioned 
models are saved in `MODEL_DIR` already. If so, it will automatically create 
a new version for the output. This allows you to non-destructively update 
the model such that you can later access both versions through TensorFlow 
Serving.

## Starting the Server

Once the build is done, you can start the server. Use the `MODEL_DIR` 
environment variable to specify the location of the models to serve. (It 
should have the same value as it did for the TensorRT conversion step.)

```shell
MODEL_DIR=/media/mars/Data/models/trt_models docker compose up -d
```

The `docker compose` project is configured to automatically launch the server 
whenever the Jetson boots. This is why you might notice high GPU usage right 
after boot.