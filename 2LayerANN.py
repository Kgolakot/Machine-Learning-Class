{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import confusion_matrix, accuracy_score, classification_report\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt"
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
    "# adding bias variable for the data x=1\n",
    "X_train = np.hstack((X_t, np.ones((X_t.shape[0], 1), dtype=X_t.dtype)))\n",
    "X_test = np.hstack((X_t1, np.ones((X_t1.shape[0], 1), dtype=X_t1.dtype))) \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train = X_train/255.0\n",
    "X_test = X_test/255.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#splitting the data into training set and validation set\n",
    "X_val = X_train[54000:60000, :]\n",
    "X_train = X_train[1:54000, :]\n",
    "\n",
    "y_val = y_train[54000:60000]\n",
    "y_train = y_train[1:54000]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#tunable hyperparameters\n",
    "epochs = 450\n",
    "learning_rate = 0.0001\n",
    "\n",
    "#losstrack\n",
    "losstrack_train = []\n",
    "losstrack_val = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#intializing bias and weights \n",
    "bias_1 = np.random.randn(1,10)*0.1\n",
    "\n",
    "weight_1 = np.random.randn(X_train.shape[1],14)*0.1\n",
    "weight_2 = np.random.randn(14,10)*0.1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#softmax function\n",
    "def softmax(z):\n",
    "    e_x = np.exp(z)\n",
    "    return e_x / e_x.sum(axis=1, keepdims =True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#sigmoid funciton\n",
    "def sigmoid(z):\n",
    "    return 1 / (1 + np.exp(-z))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def sigmoid_derivative(a):\n",
    "    p = np.multiply(a, (1-a))\n",
    "    return p"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#crossentropy function\n",
    "def cross_entropy(x, y):\n",
    "    m = y.shape[0]\n",
    "    log_likelihood = -np.log(x[range(m), y])\n",
    "    loss = np.sum(log_likelihood) / m\n",
    "    return loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def delta_weight_2(p, y, a):\n",
    "    m = y.shape[0]\n",
    "    p[range(m), y] -= 1\n",
    "    b = np.dot(a.T, p)\n",
    "    return b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#gradient of weight 1 (first layer)\n",
    "def delta_weight_1(a2, a1, w2, x, y):\n",
    "    l = np.dot(a2, w2.T)\n",
    "    q = sigmoid_derivative(a1)\n",
    "    dw1 = np.dot(x.T, q*l)\n",
    "    return dw1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#gradient of bias of second layer\n",
    "def delta_bias_1(a, y):\n",
    "    w = np.sum(a, axis = 1)\n",
    "    return w \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "for epoch in range(epochs):  \n",
    "#first layer for training set\n",
    "    z1_train = np.dot(X_train, weight_1) \n",
    "    a1_train = sigmoid(z1_train)\n",
    "#second layer of training set\n",
    "    z2_train = np.dot(a1_train, weight_2) + bias_1\n",
    "    a2_train = softmax(z2_train)\n",
    "#first layer of validation set    \n",
    "    z1_val = np.dot(X_val, weight_1) \n",
    "    a1_val = sigmoid(z1_val)\n",
    "#second layer of validation set    \n",
    "    z2_val = np.dot(a1_val, weight_2) + bias_1\n",
    "    a2_val = softmax(z2_val)\n",
    "#training loss and validation loss\n",
    "    loss_train = cross_entropy(a2_train, y_train)\n",
    "    loss_val = cross_entropy(a2_val, y_val)\n",
    "#backpropagation of gradients\n",
    "    grad_weight_2 = delta_weight_2(a2_train, y_train, a1_train)\n",
    "    grad_weight_1 = delta_weight_1(a2_train, a1_train, weight_2, X_train, y_train)\n",
    "#backpropagation of bias\n",
    "    del_bias_1 = delta_bias_1(a2_train.T, y_train)\n",
    "#gradient descent of both wieghts\n",
    "    weight_1 = weight_1 - learning_rate*grad_weight_1\n",
    "    weight_2 = weight_2 - learning_rate*grad_weight_2\n",
    "#gradient descent of bias\n",
    "    bias_1 = bias_1 - learning_rate*del_bias_1\n",
    "#plotting the loss    \n",
    "    losstrack_train.append(loss_train)\n",
    "    losstrack_val.append(loss_val)\n",
    "    plt.plot(losstrack_train, label = 'train')\n",
    "    plt.plot(losstrack_val, label = 'validation')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "z1_test = np.dot(X_test, weight_1)\n",
    "a1_test = sigmoid(z1_test)\n",
    "\n",
    "z2_test = np.dot(a1_test, weight_2) + bias_1\n",
    "a2_test = softmax(z2_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'a2_test' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-1-d665d4af4f01>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mpred\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0ma2_test\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mrange\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_test\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_test\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0mAccuracy\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msum\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpred\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m/\u001b[0m\u001b[0my_test\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m)\u001b[0m\u001b[0;34m*\u001b[0m \u001b[0;36m100\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'a2_test' is not defined"
     ]
    }
   ],
   "source": [
    "pred = a2_test[range(y_test.shape[0]), y_test]\n",
    "Accuracy = (np.sum(pred)/y_test.shape[0] )* 100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'a2_test' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-2-8f1e84c891b0>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0my_pred\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0ma2_test\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mround\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0my_pred\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0margmax\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_pred\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maxis\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0my_true\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0my_test\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'a2_test' is not defined"
     ]
    }
   ],
   "source": [
    "y_pred = a2_test.round()\n",
    "y_pred = np.argmax(y_pred, axis = 1)\n",
    "y_true = y_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "esults = confusion_matrix(y_true, y_pred) \n",
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
