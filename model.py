import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import wandb

# Initialize Weights and Biases
wandb.init(project='Github_Actions', entity='manishrai727', name = 'Github_Action_CICD')

# Log hyperparameters
config = wandb.config
config.learning_rate = 0.001
config.epochs = 100

def plot_predictions(train_data, train_labels,  test_data, test_labels,  predictions):
  plt.figure(figsize=(6, 5))

  plt.scatter(train_data, train_labels, c="b", label="Training data")
  plt.scatter(test_data, test_labels, c="g", label="Testing data")
  plt.scatter(test_data, predictions, c="r", label="Predictions")

  plt.legend()
  
  plt.title('Model Results', family='Arial', fontsize=14)
  
  plt.savefig('model_results.png', dpi=120)

def mae(y_test, y_pred):
  return tf.metrics.mean_absolute_error(y_test, y_pred)
  
def mse(y_test, y_pred):
  return tf.metrics.mean_squared_error(y_test, y_pred)

# Generate data
X = np.arange(-100, 100, 4).reshape(-1, 1)
y = np.arange(-90, 110, 4)

# Split into training and testing data
N = 25
X_train = X[:N]
y_train = y[:N]

X_test = X[N:]
y_test = y[N:]
# Reshape X_train and X_test
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

input_shape = X[0].shape 
output_shape = y[0].shape

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=input_shape), 
    tf.keras.layers.Dense(1)
    ])

# Compile the model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(learning_rate=config.learning_rate),
              metrics=['mae'])

# Train the model and log metrics to Weights and Biases
model.fit(X_train, y_train, epochs=config.epochs, 
          validation_data=(X_test, y_test), 
          callbacks=[wandb.keras.WandbCallback()])

# Make predictions on test data
y_preds = model.predict(X_test)

# Plot the results
plot_predictions(train_data=X_train, train_labels=y_train,  test_data=X_test, test_labels=y_test,  predictions=y_preds)

# Calculate and log metrics
mae_score = np.round(float(mae(y_test, y_preds.squeeze()).numpy()), 2)
mse_score = np.round(float(mse(y_test, y_preds.squeeze()).numpy()), 2)
print(f'\nMean Absolute Error = {mae_score}, Mean Squared Error = {mse_score}.')

wandb.log({'MAE': mae_score, 'MSE': mse_score})

# Save metrics to file
with open('metrics.txt', 'w') as outfile:
    outfile.write(f'\nMean Absolute Error = {mae_score}, Mean Squared Error = {mse_score}.')
