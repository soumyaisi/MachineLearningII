pip install numpy , scipy , scikit-learn , h5py , pillow

pip install Theano

pip install tensorflow

pip install keras


Installing Keras on Docker
Refer to the GitHub repository at https://github.com/saiprashanths/dl-docker for the code files.

It is also possible to access TensorBoard (for more information, refer to https://www.tensorflow.org/how_tos/summaries_and_tensorboard/ )


Installing Keras on Google Cloud ML
First, we can install Google Cloud (for the
downloadable file, refer to https://cloud.google.com/sdk/ ), a command-line interface for Google Cloud
Platform; then we can use CloudML, a managed service that enables us to easily build
machine, learning models with TensorFlow.

we simply download the Keras source from
PyPI (for the downloadable file, refer to https://pypi.Python.org/pypi/Keras/1.2.0 or later versions)



Installing Keras on Amazon AWS
Indeed, it is possible to use a prebuilt
AMI named TFAMI.v3 that is open and free (for more information, refer to https://github.com/ritchieng/tensorflow-a
ws-ami )


Installing Keras on Microsoft Azure
One way to install Keras on Azure is to install the support for Docker and then get a containerized
version of TensorFlow plus Keras. Online, it is also possible to find a detailed set of instructions on
how to install Keras and TensorFlow with Docker, but this is essentially what we have seen already
in a previous section (for more information, refer to https://blogs.msdn.microsoft.com/uk_faculty_connection/2016/09/2
6/tensorflow-on-docker-with-microsoft-azure/ ).



If you use Theano as the only backend, then Keras can run with just a click by loading a pre-built
package available on Cortana Intelligence Gallery (for more information, refer to https://gallery.cortanaintelli
gence.com/Experiment/Theano-Keras-1 ).



Saving and loading the weights and the
architecture of a model

Model architectures can be easily saved and loaded as follows:
# save as JSON json_string = model.to_json()
# save as YAML yaml_string = model.to_yaml()
# model reconstruction from JSON: from keras.models import model_from_json model = model_from_json(json_string)
Model parameters (weights) can be easily saved and loaded as follows:
from keras.models import load_model model.save('my_model.h5')
# creates a HDF5 file 'my_model.h5' del model
# deletes the existing model
# returns a compiled model
# identical to the previous one model = load_model('my_model.h5')


Callbacks for customizing the training process
The training process can be stopped when a metric has stopped improving by using an appropriate
callback :
keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
patience=0, verbose=0, mode='auto')
Loss history can be saved by defining a callback like the following:
class LossHistory(keras.callbacks.Callback):
def on_train_begin(self, logs={}):
verbose=0, callbacks=[history]) print history.losse


Checkpointing

Checkpointing is a process that saves a snapshot of the application's state at regular intervals, so the
application can be restarted from the last saved state in case of failure. This is useful during training
of deep learning models, which can often be a time-consuming task.
Some scenarios where checkpointing can be useful include the following:
If you want the ability to restart from your last checkpoint after your AWS Spot instance (for
more information, refer to http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/how-spot-instances-work.html ) or
Google preemptible virtual machine (for more information, refer to https://cloud.google.com/compute/docs
/instances/preemptible ) is unexpectedly terminated
If you want to stop training, perhaps to test your model on test data, then continue training from
the last checkpoint
If you want to retain the best version (by some metric such as validation loss) as it trains over
multiple epochs
The first and second scenarios can be handled by saving a checkpoint after each epoch, which is
handled by the default usage of the ModelCheckpoint callback. The following code illustrates how to add
checkpointing during training of your deep learning model in Keras:


from keras.callbacks import ModelCheckpoint
model.compile(optimizer="rmsprop", loss="categorical_crossentropy",
metrics=["accuracy"])
# save best model
checkpoint = ModelCheckpoint(
filepath=os.path.join(MODEL_DIR, "model-{epoch:02d}.h5"))
model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS,
validation_split=0.1, callbacks=[checkpoint])



Using TensorBoard and Keras

Keras provides a callback for saving your training and test metrics, as well as activation histograms
for the different layers in your model:
keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
write_graph=True, write_images=False)
Saved data can then be visualized with TensorBoad launched at the command line:
tensorboard --logdir=/full_path_to_your_logs


Using Quiver and Keras

In Chapter 3 , Deep Learning with ConvNets, we will discuss ConvNets, which are an advanced deep
learning technique for dealing with images. Here we give a preview of Quiver (for more information,
refer to https://github.com/jakebian/quiver ), a tool useful for visualizing ConvNets features in an interactive
way. The installation is pretty simple, and after that Quiver can be used with one single line:
pip install quiver_engine
from quiver_engine import server
server.launch(model)
This will launch the visualization at localhost:5000 .




