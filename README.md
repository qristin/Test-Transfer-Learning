## Training new model
In the folder NeuralNetwork a Keras model is trained on the data. It is network with 2 hidden layers with each 32 nodes. For now it is only trained on 10 images. To Run the model and predict the class of another image run
```
python NeuralNetwork/test.py
```

## Transfer learning with Tensorflow
Training the final layer of the google inception model with given input images and predict which class an image belongs to.

### Prerequisite

To be able to train and use the model you need to install the following:
- python 3 installed
- tensorflow installed `pip install tensorflow`

Create a folder named data. See 'data-example' for the expected structure.

### Train the model

To train the last layer of the network run the following code
```
python TransferLearning/retrain.py --image_dir <path to current dir>/data/training-data/images/ --output-graph <path to current dir>/data/training-data/models/output-graph.pb --output-labels <path to current dir>/data/training-data/models/output_labels.txt --bottleneck_dir ./data/training-data/models/bottleneck --how_many_training_steps 100
```

## Predict

```
python TransferLearning/predict.py --test-image ./data/test/vase.png --labels ./data/training-data/models/output_labels.txt --graph ./data/training-data/models/output_graph.pb
```

## Help

Depending on the OS the path sometimes is not correctly found. The tensorflow lib then writes the models in your os tmp folder.
