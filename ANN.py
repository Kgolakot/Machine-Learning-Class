{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "from sklearn.metrics import confusion_matrix, accuracy_score, classification_report\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import mnist_reader\n",
    "X_t, y_train = mnist_reader.load_mnist('data/fashion', kind='train')\n",
    "X_t1, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_images = X_train / 255.0\n",
    "test_images = X_test / 255.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = tf.keras.models.Sequential([\n",
    "  tf.keras.layers.Dense(128, activation = 'relu'),\n",
    "  tf.keras.layers.Dropout(0.1),\n",
    "  tf.keras.layers.Dense(64, activation='relu'),\n",
    "  tf.keras.layers.Dropout(0.1),\n",
    "  tf.keras.layers.Dense(10, activation='softmax')\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer='sgd',\n",
    "              loss='sparse_categorical_crossentropy',\n",
    "              metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "history = model.fit(train_images, y_train, epochs=30, validation_data=(test_images, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.plot(history.history['acc'], label='accuracy')\n",
    "plt.plot(history.history['val_acc'], label = 'val_accuracy')\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.ylim([0.5, 1])\n",
    "plt.legend(loc='lower right')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.plot(history.history['loss'], label='training loss')\n",
    "plt.plot(history.history['val_loss'], label = 'validation loss')\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('loss')\n",
    "plt.legend(loc='upper right')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_loss, test_acc = model.evaluate(test_images,  y_test, verbose=2)\n",
    "print('\\nTest accuracy:', test_acc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions = model.predict(test_images)\n",
    "\n",
    "y_pred = np.argmax(predictions, axis =1)\n",
    "y_true = y_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "results = confusion_matrix(y_true, y_pred) \n",
    "print 'Confusion Matrix :'\n",
    "print(results) \n",
    "print 'Accuracy Score :',accuracy_score(y_true, y_pred) \n",
    "print 'Report : '\n",
    "print classification_report(y_true, y_pred)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
