{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.cluster import KMeans\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def cluster_labels(model, true_labels):\n",
    "\n",
    "    cluster_labels = {}\n",
    "\n",
    "    for i in range(10):\n",
    "\n",
    "        # find index of points in cluster\n",
    "        labels = []\n",
    "        index = np.where(model== i)\n",
    "\n",
    "        # append actual labels for each point in cluster\n",
    "        labels.append(true_labels[index])\n",
    "\n",
    "        # determine most common label\n",
    "        if len(labels[0]) == 1:\n",
    "            counts = np.bincount(labels[0])\n",
    "        else:\n",
    "            counts = np.bincount(np.squeeze(labels))\n",
    "\n",
    "        # assign the cluster to a value in the cluster_labels dictionary\n",
    "        if np.argmax(counts) in cluster_labels:\n",
    "            # append the new number to the existing array at this index\n",
    "            cluster_labels[np.argmax(counts)].append(i)\n",
    "        else:\n",
    "            # create a new array in this indeex\n",
    "            cluster_labels[np.argmax(counts)] = [i]\n",
    "\n",
    "        \n",
    "    return cluster_labels  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def infer_labels(X_labels, cluster_labels):\n",
    "    #assign labels to the predicted data according to the model\n",
    "    # empty array of len(X)\n",
    "    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)\n",
    "    \n",
    "    for i, cluster in enumerate(X_labels):\n",
    "        for key, value in cluster_labels.items():\n",
    "            if cluster in value:\n",
    "                predicted_labels[i] = key\n",
    "                \n",
    "    return predicted_labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import mnist_reader\n",
    "X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')\n",
    "X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')\n",
    "\n",
    "X_train, X_test = X_train/255. , X_test/255."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Define the K-means cluster\n",
    "km = KMeans(n_clusters = 10, n_init = 20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#fit the model\n",
    "fit_model = km.fit(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#predict the clusters using the data\n",
    "test_clusters = fit_model.predict(X_test)\n",
    "\n",
    "#assign the cluster labels based on target data\n",
    "kmeans_labels = cluster_labels(test_clusters, y_test)\n",
    "\n",
    "#infer the predicted labels from the model\n",
    "predicted_train_labels = infer_labels(test_clusters, kmeans_labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Test the accuracy\n",
    "Accuracy = accuracy_score(y_test, predicted_train_labels)\n",
    "\n",
    "print(\"Accuracy of K-means model is \", Accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#reshape data in to 4 dimensional array\n",
    "X_train = X_train.reshape(-1, 28,28, 1)\n",
    "X_test = X_test.reshape(-1, 28,28, 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#split the data into training and test data\n",
    "from sklearn.model_selection import train_test_split\n",
    "train_X,valid_X,train_ground,valid_ground = train_test_split(X_train, X_train, test_size=0.2, random_state=13)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras import Model,Sequential\n",
    "from keras.layers import Flatten,Conv2D,Dense,Reshape,Conv2DTranspose"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def auto_encoder(input_shape=(28, 28, 1), filters=[32, 64, 128, 10]):\n",
    "    model = Sequential()\n",
    "    if input_shape[0] % 8 == 0:\n",
    "        pad3 = 'same'\n",
    "    else:\n",
    "        pad3 = 'valid'\n",
    "    #encoder    \n",
    "    model.add(Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))\n",
    "\n",
    "    model.add(Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2'))\n",
    "\n",
    "    model.add(Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3'))\n",
    "\n",
    "    model.add(Flatten())\n",
    "    #latent space\n",
    "    model.add(Dense(units=filters[3], name='latent_space'))\n",
    "    #decoder\n",
    "    model.add(Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu'))\n",
    "\n",
    "    model.add(Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2])))\n",
    "    model.add(Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))\n",
    "\n",
    "    model.add(Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2'))\n",
    "\n",
    "    model.add(Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1'))\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#assign the function to the model\n",
    "autoencoder = auto_encoder()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#autoencoder summary\n",
    "autoencoder.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#compile the autoencoder\n",
    "autoencoder.compile(optimizer='RMSprop', loss='mse')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#train the model on the using the split training data and evaluate the loss for training and validation\n",
    "model = autoencoder.fit(train_X, train_ground, batch_size=256,epochs=100, validation_data = (valid_X, valid_ground))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from matplotlib import pyplot as plt\n",
    "#plot the training loss and valisation loss\n",
    "loss = model.history['loss']\n",
    "val_loss = model.history['val_loss']\n",
    "epochs = range(100)\n",
    "plt.figure()\n",
    "plt.plot(epochs, loss, 'bo', label='Training loss')\n",
    "plt.plot(epochs, val_loss, 'b', label='Validation loss')\n",
    "plt.title('Training and validation loss')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#create the encoder model with output as latent space\n",
    "intermediate_layer_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('latent_space').output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#encoder summary\n",
    "intermediate_layer_model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#find the latent space representaion of training data and test data\n",
    "latent_space = intermediate_layer_model.predict(X_train)\n",
    "latent_space_test = intermediate_layer_model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#create new k-means model for encoded layer with 10 clusters\n",
    "nm = KMeans(n_clusters = 10, n_init=50)\n",
    "#fit the model for the latent space respresentation of training data\n",
    "new_model = nm.fit(latent_space)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#predict the clusters for the training and test latent space\n",
    "latent_clusters = new_model.predict(latent_space)\n",
    "latent_clusters_test = new_model.predict(latent_space_test)\n",
    "#infer the cluster labels of the model based on the target labels \n",
    "latent_labels = cluster_labels(latent_clusters, y_train)\n",
    "latent_labels_test = cluster_labels(latent_clusters_test, y_test)\n",
    "#assign the cluster labels to the data \n",
    "predicted_new_labels = infer_labels(latent_clusters, latent_labels)\n",
    "predicted_new_labels_test = infer_labels(latent_clusters_test, latent_labels_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "Accuracy_train = accuracy_score(y_train, predicted_new_labels)\n",
    "Accuracy_test = accuracy_score(y_test, predicted_new_labels_test)\n",
    "\n",
    "print(\"Training Accuracy of convolution autoencoder with k_memans is\", Accuracy_train)\n",
    "print(\"Test Accuracy of convolution autoencoder with k_memans is\", Accuracy_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.mixture import GaussianMixture"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#create a gaussian mixture model\n",
    "gmm = GaussianMixture(n_components = 10)\n",
    "#fit the gaussian mixture model\n",
    "g_model = gmm.fit(latent_space)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#predict the gaussian mixture model clusters based on the latent space representation of the test data and training data\n",
    "gmm_clusters = g_model.predict(latent_space)\n",
    "gmm_clusters_test = g_model.predict(latent_space_test)\n",
    "#infer the cluster labels of the gmm model based on the target data\n",
    "gmm_labels = cluster_labels(gmm_clusters, y_train)\n",
    "gmm_labels_test = cluster_labels(gmm_clusters_test, y_test)\n",
    "#assign the cluster labels to the predicted data\n",
    "pred_gmm = infer_labels(gmm_clusters, gmm_labels)\n",
    "pred_gmm_test = infer_labels(gmm_clusters_test, gmm_labels_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "Accuracy_train = accuracy_score(y_train, pred_gmm)\n",
    "Accuracy_test = accuracy_score(y_test, pred_gmm_test)\n",
    "\n",
    "print(\"Training Accuracy of convolution autoencoder with gaussian_mixture_model is\",Accuracy_train)\n",
    "print(\"Test Accuracy of convolution autoencoder with gaussian_mixture_model is\",Accuracy_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "#construct confusion matrix for the kmeans model and gaussian mixture model\n",
    "cm_kmeans = confusion_matrix(y_test,predicted_new_labels_test)\n",
    "cm_gmm = confusion_matrix(y_test, pred_gmm_test)\n",
    "\n",
    "print(\"confusion matrix for encoded k-means model \\n\",cm_kmeans)\n",
    "print(\"confusion matrix for encoded gmm model \\n\",cm_gmm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import seaborn as sn\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "km_cm = pd.DataFrame(cm_kmeans, index = [i for i in \"ABCDEFGHIJ\"],\n",
    "                  columns = [i for i in \"ABCDEFGHIJ\"])\n",
    "plt.figure(figsize = (10,7))\n",
    "sn.heatmap(km_cm, annot=True)\n",
    "\n",
    "gmm_cm = pd.DataFrame(cm_gmm, index = [i for i in \"ABCDEFGHIJ\"],\n",
    "                  columns = [i for i in \"ABCDEFGHIJ\"])\n",
    "plt.figure(figsize = (10,7))\n",
    "sn.heatmap(gmm_cm, annot=True)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
