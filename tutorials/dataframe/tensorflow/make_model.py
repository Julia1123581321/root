import numpy as np
import tensorflow as tf
from keras.layers.core import Dense
from keras.models import Sequential
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    
# To test the TensorflowCEvaluator, we need to build a model first
# Let's make an XOR gate using a sequential model
model = Sequential()
# This number of dense layers and input neurons shows good fitting performance later on
model.add(Dense(32, input_dim=2, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer="adam", loss="mean_squared_error", metrics=['binary_accuracy'])

# The training data corresponds to the input and output logic of the XOR gate
x_train = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
y_train = np.array([[0],[1],[1],[0]], "float32")

# Since we want a "working" model, we need to train it first on our training data over several epochs
model.fit(x_train, y_train, epochs=100, verbose=2)

# Now we that we have a trained model, we want to save it in a pb file as a "frozen graph"
# These steps are necessary to make the model readable by the TensorflowCEvaluator
# Note that simply saving the model does not provide the right structure for the model to be read.
# So we need to convert our Keras model into a TF Graph using tf.function
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# To produce a single pb file containing the full model including variables and hyperparameters,
# we need to "freeze" our model
frozen_func = convert_variables_to_constants_v2(full_model,lower_control_flow=False)
frozen_func.graph.as_graph_def()
layers = [op.name for op in frozen_func.graph.get_operations()]
# The last step is to save the frozen graph from the frozen Concrete Function to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="frozen_graph.pb",
                  as_text=False)
