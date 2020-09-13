{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import confusion_matrix, accuracy_score, classification_report\n",
    "from tensorflow.keras import layers, models\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Read Fashion MNIST dataset\n",
    "\n",
    "import mnist_reader\n",
    "\n",
    "X_t, y_train = mnist_reader.load_mnist('data/fashion', kind='train')\n",
    "X_tt, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_t = X_t/ 255.0\n",
    "X_tt = X_tt/255.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train = np.reshape(X_t, (60000, 28, 28, 1))\n",
    "X_test = np.reshape(X_tt, (10000, 28, 28, 1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_images, test_images = X_train, X_test "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = models.Sequential()\n",
    "model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28 ,1)))\n",
    "model.add(layers.MaxPooling2D((2,2)))\n",
    "model.add(layers.Conv2D(64, (3,3), activation='relu'))\n",
    "model.add(layers.MaxPooling2D((2,2)))\n",
    "model.add(layers.Flatten())\n",
    "model.add(layers.Dense(128, activation='relu'))\n",
    "model.add(layers.Dense(10, activation='softmax'))"
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
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "batch_size = 128\n",
    "\n",
    "history = model.fit(train_images, y_train, batch_size = batch_size, epochs = 20, verbose = 1,\n",
    "                    validation_data=(test_images, y_test))"
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
    "print('\\nTest accuracy:', test_acc)\n"
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
