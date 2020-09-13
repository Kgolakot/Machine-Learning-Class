{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled1.ipynb",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "IUiyTJDkc1_Q",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import gym\n",
        "import gym.spaces\n",
        "import time\n",
        "import copy\n",
        "import threading\n",
        "import time\n",
        "import collections"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "r9-jw2V0dBTi",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "class GridEnvironment(gym.Env):\n",
        "    metadata = { 'render.modes': ['human'] }\n",
        "    \n",
        "    def __init__(self, normalize=False, size=4):\n",
        "        self.observation_space = gym.spaces.Box(0, size, (size,))\n",
        "        self.action_space = gym.spaces.Discrete(4)\n",
        "        self.max_timesteps = size*2 + 1\n",
        "        self.normalize = normalize\n",
        "        self.size = size\n",
        "\n",
        "        # Generate State Transition Table\n",
        "        self.transition_matrix = []\n",
        "        for x in range(size + 1):\n",
        "            state_x = []\n",
        "            for y in range(size + 1):\n",
        "                state_y = []\n",
        "                for a in range(4):\n",
        "                    one_hot = np.zeros(4)\n",
        "                    one_hot[a] = 1\n",
        "                    state_y.append(one_hot)\n",
        "                state_x.append(state_y)\n",
        "            self.transition_matrix.append(state_x)\n",
        "        \n",
        "    def transition_func(self, x, y, action, return_probs=False):\n",
        "        probs = self.transition_matrix[x][y][action]\n",
        "        if return_probs:\n",
        "            return probs\n",
        "        else:\n",
        "            return np.random.choice(len(probs), p=probs)\n",
        "\n",
        "    def _get_distance(self, x, y):\n",
        "        return abs(x[0] - y[0]) + abs(x[1] - y[1])\n",
        "        \n",
        "    def reset(self):\n",
        "        self.timestep = 0\n",
        "        self.agent_pos = [0, 0]\n",
        "        self.goal_pos = [self.size, self.size]\n",
        "        self.state = np.zeros((self.size + 1, self.size + 1))\n",
        "        self.state[tuple(self.agent_pos)] = 1\n",
        "        self.state[tuple(self.goal_pos)] = 0.5\n",
        "        self.prev_distance = self._get_distance(self.agent_pos, self.goal_pos)\n",
        "        return np.array(self.agent_pos)/1.\n",
        "    \n",
        "    def step(self, action):\n",
        "        action_taken = self.transition_func(self.agent_pos[0], self.agent_pos[1], action)\n",
        "        self.state = np.random.choice(self.observation_space.shape[0])\n",
        "        if action_taken == 0:\n",
        "            self.agent_pos[0] += 1\n",
        "        if action_taken == 1:\n",
        "            self.agent_pos[0] -= 1\n",
        "        if action_taken == 2:\n",
        "            self.agent_pos[1] += 1\n",
        "        if action_taken == 3:\n",
        "            self.agent_pos[1] -= 1\n",
        "          \n",
        "        self.agent_pos = np.clip(self.agent_pos, 0, self.size)\n",
        "        self.state = np.zeros((self.size + 1, self.size + 1))\n",
        "        self.state[tuple(self.agent_pos)] = 1\n",
        "        self.state[tuple(self.goal_pos)] = 0.5\n",
        "        \n",
        "        current_distance = self._get_distance(self.agent_pos, self.goal_pos)\n",
        "        if current_distance < self.prev_distance:\n",
        "            reward = 1\n",
        "        elif current_distance > self.prev_distance:\n",
        "            reward = -1\n",
        "        else:\n",
        "            reward = -1\n",
        "        self.prev_distance = current_distance\n",
        "        \n",
        "        self.timestep += 1\n",
        "        if self.timestep >= self.max_timesteps or current_distance == 0:\n",
        "            done = True\n",
        "        else:\n",
        "            done = False\n",
        "        info = {}\n",
        "        \n",
        "        obs = self.agent_pos\n",
        "        if self.normalize:\n",
        "            obs = obs/self.size\n",
        "        return obs, reward, done, info\n",
        "        \n",
        "    def render(self, mode='human'):\n",
        "        plt.imshow(self.state)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "VR7G8JeTdIMZ",
        "colab_type": "code",
        "outputId": "f56cedf9-0ce8-4248-922b-bf617b256c89",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 265
        }
      },
      "source": [
        "env = GridEnvironment()\n",
        "obs = env.reset()\n",
        "env.render()"
      ],
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAPUAAAD4CAYAAAA0L6C7AAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjIsIGh0\ndHA6Ly9tYXRwbG90bGliLm9yZy8li6FKAAAIxElEQVR4nO3dz4uchR3H8c+nmzUxWJDWHDQbGg8i\nBKEJLCGQW0CMP9CrAT0Je6kQQRA9+gfUevESNFhQFEEPEiwh1IgINnETYzBGJYjFiLC2IppCExM/\nPexQUslmnpk8zzw7375fsLCzM8x8CPvOM/PsMuskAlDHr/oeAKBdRA0UQ9RAMUQNFEPUQDFrurjT\nm34zk82bZru469Z9fnJ93xOAkf1b/9KFnPeVrusk6s2bZnX04KYu7rp1d92yte8JwMiO5K8rXsfT\nb6AYogaKIWqgGKIGiiFqoBiiBoohaqAYogaKIWqgGKIGiiFqoBiiBoohaqAYogaKIWqgGKIGiiFq\noJhGUdvebfsz22dsP9n1KADjGxq17RlJz0m6W9IWSXtsb+l6GIDxNDlSb5d0JskXSS5IelXSA93O\nAjCuJlFvlPTVZZfPDr72P2wv2F60vfjtPy+1tQ/AiFo7UZZkX5L5JPMbfjvT1t0CGFGTqL+WdPn7\n/c4NvgZgFWoS9QeSbrN9q+3rJD0o6c1uZwEY19A3809y0fajkg5KmpG0P8mpzpcBGEujv9CR5C1J\nb3W8BUAL+I0yoBiiBoohaqAYogaKIWqgGKIGiiFqoBiiBoohaqAYogaKIWqgGKIGiiFqoBiiBooh\naqAYogaKafQmCaP6/OR63XXL1i7uGsAQHKmBYogaKIaogWKIGiiGqIFiiBoohqiBYogaKIaogWKI\nGiiGqIFiiBoohqiBYogaKIaogWKIGiiGqIFihkZte7/tJdsfT2IQgGvT5Ej9oqTdHe8A0JKhUSd5\nV9J3E9gCoAW8pgaKae3dRG0vSFqQpHVa39bdAhhRa0fqJPuSzCeZn9Xatu4WwIh4+g0U0+RHWq9I\nel/S7bbP2n6k+1kAxjX0NXWSPZMYAqAdPP0GiiFqoBiiBoohaqAYogaKIWqgGKIGiiFqoBiiBooh\naqAYogaKIWqgGKIGiiFqoBiiBoohaqAYogaKIWqgGKIGiiFqoBiiBoohaqAYogaKIWqgGKIGiiFq\noBiiBoohaqAYogaKIWqgGKIGiiFqoBiiBoohaqAYogaKGRq17U22D9v+xPYp23snMQzAeNY0uM1F\nSY8nOW7715KO2T6U5JOOtwEYw9AjdZJvkhwffP6jpNOSNnY9DMB4mhyp/8v2ZknbJB25wnULkhYk\naZ3WtzANwDganyizfYOk1yU9luSHX16fZF+S+STzs1rb5kYAI2gUte1ZLQf9cpI3up0E4Fo0Oftt\nSS9IOp3kme4nAbgWTY7UOyU9LGmX7RODj3s63gVgTENPlCV5T5InsAVAC/iNMqAYogaKIWqgGKIG\niiFqoBiiBoohaqAYogaKIWqgGKIGiiFqoBiiBoohaqAYogaKIWqgGKIGiiFqoBiiBoohaqAYogaK\nIWqgGKIGiiFqoBiiBoohaqAYogaKIWqgGKIGiiFqoBiiBoohaqAYogaKIWqgGKIGihkate11to/a\n/sj2KdtPT2IYgPGsaXCb85J2JTlne1bSe7b/kuRvHW8DMIahUSeJpHODi7ODj3Q5CsD4Gr2mtj1j\n+4SkJUmHkhzpdhaAcTWKOsmlJFslzUnabvuOX97G9oLtRduLP+l82zsBNDTS2e8k30s6LGn3Fa7b\nl2Q+yfys1ra1D8CImpz93mD7xsHn10u6U9KnXQ8DMJ4mZ79vlvRn2zNa/k/gtSQHup0FYFxNzn6f\nlLRtAlsAtIDfKAOKIWqgGKIGiiFqoBiiBoohaqAYogaKIWqgGKIGiiFqoBiiBoohaqAYogaKIWqg\nGKIGiiFqoJgm73wC/F8486cdfU9o7PwfV37bfY7UQDFEDRRD1EAxRA0UQ9RAMUQNFEPUQDFEDRRD\n1EAxRA0UQ9RAMUQNFEPUQDFEDRRD1EAxRA0UQ9RAMUQNFNM4atsztj+0faDLQQCuzShH6r2STnc1\nBEA7GkVte07SvZKe73YOgGvV9Ej9rKQnJP280g1sL9hetL34k863Mg7A6IZGbfs+SUtJjl3tdkn2\nJZlPMj+rta0NBDCaJkfqnZLut/2lpFcl7bL9UqerAIxtaNRJnkoyl2SzpAclvZ3koc6XARgLP6cG\nihnpz+4keUfSO50sAdAKjtRAMUQNFEPUQDFEDRRD1EAxRA0UQ9RAMUQNFEPUQDFEDRRD1EAxRA0U\nQ9RAMUQNFEPUQDFEDRTjJO3fqf2tpL+3fLc3SfpHy/fZpWnaO01bpena29XW3yXZcKUrOom6C7YX\nk8z3vaOpado7TVul6drbx1aefgPFEDVQzDRFva/vASOapr3TtFWarr0T3zo1r6kBNDNNR2oADRA1\nUMxURG17t+3PbJ+x/WTfe67G9n7bS7Y/7nvLMLY32T5s+xPbp2zv7XvTSmyvs33U9keDrU/3vakJ\n2zO2P7R9YFKPueqjtj0j6TlJd0vaImmP7S39rrqqFyXt7ntEQxclPZ5ki6Qdkv6wiv9tz0valeT3\nkrZK2m17R8+bmtgr6fQkH3DVRy1pu6QzSb5IckHLf3nzgZ43rSjJu5K+63tHE0m+SXJ88PmPWv7m\n29jvqivLsnODi7ODj1V9ltf2nKR7JT0/ycedhqg3SvrqsstntUq/8aaZ7c2Stkk60u+SlQ2eyp6Q\ntCTpUJJVu3XgWUlPSPp5kg86DVGjY7ZvkPS6pMeS/ND3npUkuZRkq6Q5Sdtt39H3ppXYvk/SUpJj\nk37saYj6a0mbLrs8N/gaWmB7VstBv5zkjb73NJHke0mHtbrPXeyUdL/tL7X8knGX7Zcm8cDTEPUH\nkm6zfavt67T8h+/f7HlTCbYt6QVJp5M80/eeq7G9wfaNg8+vl3SnpE/7XbWyJE8lmUuyWcvfs28n\neWgSj73qo05yUdKjkg5q+UTOa0lO9btqZbZfkfS+pNttn7X9SN+brmKnpIe1fBQ5Mfi4p+9RK7hZ\n0mHbJ7X8H/2hJBP7MdE04ddEgWJW/ZEawGiIGiiGqIFiiBoohqiBYogaKIaogWL+Ax8V0jpegaxN\nAAAAAElFTkSuQmCC\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wPSzs6wYdLly",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "class RandomAgent:\n",
        "    def __init__(self, env):\n",
        "        self.env = env\n",
        "        self.observation_space = env.observation_space\n",
        "        self.action_space = env.action_space\n",
        "\n",
        "    def policy(self, observation):\n",
        "        return np.random.choice(self.action_space.n)\n",
        "        \n",
        "    def step(self, observation, verbose=False):\n",
        "        return self.policy(observation)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "rBVG34HkdOT_",
        "colab_type": "code",
        "outputId": "4f634707-6355-47c2-d27c-e18423057ad8",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        }
      },
      "source": [
        "env = GridEnvironment(normalize=True)\n",
        "agent = RandomAgent(env)\n",
        "\n",
        "obs = env.reset()\n",
        "done = False\n",
        "agent.epsilon = 0\n",
        "env.render()\n",
        "plt.show()\n",
        "\n",
        "while not done:\n",
        "    action = agent.step(obs, verbose=True)\n",
        "    obs, reward, done, info = env.step(action)\n",
        "    env.render()\n",
        "    plt.show()"
      ],
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\nAAAAAElFTkSuQmCC\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\nI9FysxyeJwAAAABJRU5ErkJggg==\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\nGNFyrGAbBQAAAABJRU5ErkJggg==\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\nTkSuQmCC\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\nRK5CYII=\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\ntHeatkrTtbePrTz9BoohaqCYaYp6qe8BI5qmvdO0VZquvRPfOjXfUwNoZprO1AAaIGqgmKmI2vaC\n7c9tn7b9ZN97rsT2fttnbX/a95ZhbG+xfdj2SdsnbO/re9NabG+0/YHtjwdbn+57UxO2Z2x/ZPvA\npB5z3Udte0bSc5LulrRN0l7b2/pddUUvSlroe0RDFyU9nmSbpF2S/riO/21XJO1J8jtJ2yUt2N7V\n86Ym9kk6NckHXPdRS9op6XSSL5Kc1+pf3nyg501rSvKupG/73tFEkm+SHBu8/4NWv/g297vq8rLq\n3ODm7OBtXV/ltT0n6V5Jz0/ycach6s2Svrrk9hmt0y+8aWZ7q6Qdko70u2Rtg6eyxyWdlXQoybrd\nOvCspCck/TzJB52GqNEx2zdKel3SY0m+73vPWpL8lGS7pDlJO23f0femtdi+T9LZJEcn/djTEPXX\nkrZccntu8DG0wPasVoN+Ockbfe9pIsl3kg5rfV+72C3pfttfavVbxj22X5rEA09D1B9Kus32rbav\n1eofvn+z500l2LakFySdSvJM33uuxPYm2zcN3r9O0p2SPut31dqSPJVkLslWrX7Nvp3koUk89rqP\nOslFSY9KOqjVCzmvJTnR76q12X5F0vuSbrd9xvYjfW+6gt2SHtbqWeT44O2evket4WZJh21/otX/\n6A8lmdiPiaYJvyYKFLPuz9QARkPUQDFEDRRD1EAxRA0UQ9RAMUQNFPNfKS/OoQi9o6cAAAAASUVO\nRK5CYII=\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n1F2wvZBkvu8dTU3T3mnaKk3X3j62cvcbKIaogWKmKeo9fQ8Y0TTtnaat0nTtnfjWqXlMDaCZaTpT\nA2iAqIFipiJq2ztsf2b7lO0n+95zJbb32j5j+5O+twxje73tg7ZP2D5ue3ffm5Zje43tD2x/PNj6\ndN+bmrA9Y/sj2/smdZsrPmrbM5Kek3S3pI2Sdtre2O+qK3pR0o6+RzR0QdLjSTZK2irp9yv433ZR\n0vYkv5G0SdIO21t73tTEbkknJ3mDKz5qSVsknUryeZJzWvrLmw/0vGlZSd6V9G3fO5pI8nWSI4PP\nf9DSD9+6flddXpacHVycHXys6Gd5bc9JulfS85O83WmIep2kLy+5fFor9AdvmtneIGmzpEP9Llne\n4K7sUUlnJB1IsmK3Djwr6QlJP07yRqchanTM9g2SXpf0WJLv+96znCQXk2ySNCdpi+3b+960HNv3\nSTqT5PCkb3saov5K0vpLLs8NvoYW2J7VUtAvJ3mj7z1NJPlO0kGt7Ocutkm63/YXWnrIuN32S5O4\n4WmI+kNJt9q+xfYqLf3h+zd73lSCbUt6QdLJJM/0vedKbK+1fePg82sl3Snp035XLS/JU0nmkmzQ\n0s/s20kemsRtr/iok1yQ9Kik/Vp6Iue1JMf7XbU8269Iel/SbbZP236k701XsE3Sw1o6ixwdfNzT\n96hl3CTpoO1jWvqP/kCSib1MNE34NVGgmBV/pgYwGqIGiiFqoBiiBoohaqAYogaKIWqgmP8C+HDV\njKyKl5oAAAAASUVORK5CYII=\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\nac7DAAAAAElFTkSuQmCC\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "IU8IdTNbdRM1",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "class HeuristicAgent:\n",
        "    def __init__(self, env):\n",
        "        self.env = env\n",
        "        self.observation_space = env.observation_space\n",
        "        self.action_space = env.action_space\n",
        "\n",
        "    def policy(self, observation):\n",
        "        # 0 - down\n",
        "        # 1 - up\n",
        "        # 2 - right\n",
        "        # 3 - left\n",
        "        if (observation[0] < 1.):\n",
        "            return 0\n",
        "        if (observation[1] < 1.):\n",
        "            return 2\n",
        "        return 0\n",
        "        \n",
        "    def step(self, observation, verbose=False):\n",
        "        if verbose:\n",
        "            print(observation)\n",
        "        return self.policy(observation)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1B_pwcO7dUyh",
        "colab_type": "code",
        "outputId": "0a4d56bf-630a-457c-fa26-3092b5ec2ec3",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        }
      },
      "source": [
        "env = GridEnvironment(normalize=True)\n",
        "agent = HeuristicAgent(env)\n",
        "\n",
        "obs = env.reset()\n",
        "done = False\n",
        "agent.epsilon = 0\n",
        "env.render()\n",
        "plt.show()\n",
        "\n",
        "while not done:\n",
        "    action = agent.step(obs, verbose=True)\n",
        "    obs, reward, done, info = env.step(action)\n",
        "    env.render()\n",
        "    plt.show()"
      ],
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\nAAAAAElFTkSuQmCC\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "stream",
          "text": [
            "[0. 0.]\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\njLoY2OEAAAAASUVORK5CYII=\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "stream",
          "text": [
            "[0.25 0.  ]\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n1F2wvZBkvu8dTU3T3mnaKk3X3j62cvcbKIaogWKmKeo9fQ8Y0TTtnaat0nTtnfjWqXlMDaCZaTpT\nA2iAqIFipiJq2ztsf2b7lO0n+95zJbb32j5j+5O+twxje73tg7ZP2D5ue3ffm5Zje43tD2x/PNj6\ndN+bmrA9Y/sj2/smdZsrPmrbM5Kek3S3pI2Sdtre2O+qK3pR0o6+RzR0QdLjSTZK2irp9yv433ZR\n0vYkv5G0SdIO21t73tTEbkknJ3mDKz5qSVsknUryeZJzWvrLmw/0vGlZSd6V9G3fO5pI8nWSI4PP\nf9DSD9+6flddXpacHVycHXys6Gd5bc9JulfS85O83WmIep2kLy+5fFor9AdvmtneIGmzpEP9Llne\n4K7sUUlnJB1IsmK3Djwr6QlJP07yRqchanTM9g2SXpf0WJLv+96znCQXk2ySNCdpi+3b+960HNv3\nSTqT5PCkb3saov5K0vpLLs8NvoYW2J7VUtAvJ3mj7z1NJPlO0kGt7Ocutkm63/YXWnrIuN32S5O4\n4WmI+kNJt9q+xfYqLf3h+zd73lSCbUt6QdLJJM/0vedKbK+1fePg82sl3Snp035XLS/JU0nmkmzQ\n0s/s20kemsRtr/iok1yQ9Kik/Vp6Iue1JMf7XbU8269Iel/SbbZP236k701XsE3Sw1o6ixwdfNzT\n96hl3CTpoO1jWvqP/kCSib1MNE34NVGgmBV/pgYwGqIGiiFqoBiiBoohaqAYogaKIWqgmP8C+HDV\njKyKl5oAAAAASUVORK5CYII=\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "stream",
          "text": [
            "[0.5 0. ]\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAPUAAAD4CAYAAAA0L6C7AAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjIsIGh0\ndHA6Ly9tYXRwbG90bGliLm9yZy8li6FKAAAIx0lEQVR4nO3dz4uchR3H8c+nmzUxWhDaHDQbGg9W\nCGITWNJAbili/IFeDehJ2EuFCILo0T+g1ouXoMGCogh6kGAJoUZEsNFNjMEkWoJYjAixFdG0NDHx\n08MOJZVs5pnJ88yz8+X9goWdneWZD2HfeWaeXXadRADq+FnfAwC0i6iBYogaKIaogWKIGihmVRcH\nvcars0bXdXFoAJL+o3/pfM75cvd1EvUaXaff+nddHBqApEP5y7L38fQbKIaogWKIGiiGqIFiiBoo\nhqiBYogaKIaogWKIGiiGqIFiiBoohqiBYogaKIaogWKIGiiGqIFiiBooplHUtnfa/tT2KdtPdD0K\nwPiGRm17RtKzku6StEnSLtubuh4GYDxNztRbJZ1K8lmS85JekXR/t7MAjKtJ1OslfXHJ7dODj/0f\n2wu2F20v/qBzbe0DMKLWLpQl2ZNkPsn8rFa3dVgAI2oS9ZeSNlxye27wMQArUJOoP5B0i+2bbV8j\n6QFJb3Q7C8C4hv4y/yQXbD8iab+kGUl7kxzvfBmAsTT6Cx1J3pT0ZsdbALSAnygDiiFqoBiiBooh\naqAYogaKIWqgGKIGiiFqoBiiBoohaqAYogaKIWqgGKIGiiFqoBiiBoohaqAYogaKIWqgGKIGiiFq\noBiiBoohaqAYogaKIWqgGKIGiiFqoBiiBoohaqAYogaKIWqgGKIGiiFqoBiiBoohaqCYoVHb3mv7\njO2PJzEIwNVpcqZ+QdLOjncAaMnQqJO8I+mbCWwB0AJeUwPFrGrrQLYXJC1I0hqtbeuwAEbU2pk6\nyZ4k80nmZ7W6rcMCGBFPv4FimnxL62VJ70m61fZp2w93PwvAuIa+pk6yaxJDALSDp99AMUQNFEPU\nQDFEDRRD1EAxRA0UQ9RAMUQNFEPUQDFEDRRD1EAxRA0UQ9RAMUQNFEPUQDFEDRRD1EAxRA0UQ9RA\nMUQNFEPUQDFEDRRD1EAxRA0UQ9RAMUQNFEPUQDFEDRRD1EAxRA0UQ9RAMUQNFEPUQDFEDRRD1EAx\nQ6O2vcH2QdsnbB+3vXsSwwCMZ1WDz7kg6bEkR2z/XNJh2weSnOh4G4AxDD1TJ/kqyZHB+99LOilp\nfdfDAIynyZn6f2xvlLRF0qHL3LcgaUGS1mhtC9MAjKPxhTLb10t6TdKjSb776f1J9iSZTzI/q9Vt\nbgQwgkZR257VUtAvJXm920kArkaTq9+W9Lykk0me7n4SgKvR5Ey9XdJDknbYPjp4u7vjXQDGNPRC\nWZJ3JXkCWwC0gJ8oA4ohaqAYogaKIWqgGKIGiiFqoBiiBoohaqAYogaKIWqgGKIGiiFqoBiiBooh\naqAYogaKIWqgmJF+m2hTv77939q//2gXh27dnTdt7nsC0CrO1EAxRA0UQ9RAMUQNFEPUQDFEDRRD\n1EAxRA0UQ9RAMUQNFEPUQDFEDRRD1EAxRA0UQ9RAMUQNFEPUQDFDo7a9xvb7tj+yfdz2U5MYBmA8\nTX6d0TlJO5KctT0r6V3bf07y1463ARjD0KiTRNLZwc3ZwVu6HAVgfI1eU9uesX1U0hlJB5Ic6nYW\ngHE1ijrJxSSbJc1J2mr7tp9+ju0F24u2F7/+58W2dwJoaKSr30m+lXRQ0s7L3LcnyXyS+XW/mGlr\nH4ARNbn6vc72DYP3r5V0h6RPuh4GYDxNrn7fKOlPtme09J/Aq0n2dTsLwLiaXP0+JmnLBLYAaAE/\nUQYUQ9RAMUQNFEPUQDFEDRRD1EAxRA0UQ9RAMUQNFEPUQDFEDRRD1EAxRA0UQ9RAMUQNFEPUQDFN\nfvPJyP52bK3uvGlzF4cGOnPqj9v6ntDYuT8s/2v3OVMDxRA1UAxRA8UQNVAMUQPFEDVQDFEDxRA1\nUAxRA8UQNVAMUQPFEDVQDFEDxRA1UAxRA8UQNVAMUQPFEDVQTOOobc/Y/tD2vi4HAbg6o5ypd0s6\n2dUQAO1oFLXtOUn3SHqu2zkArlbTM/Uzkh6X9ONyn2B7wfai7cUfdK6VcQBGNzRq2/dKOpPk8JU+\nL8meJPNJ5me1urWBAEbT5Ey9XdJ9tj+X9IqkHbZf7HQVgLENjTrJk0nmkmyU9ICkt5I82PkyAGPh\n+9RAMSP92Z0kb0t6u5MlAFrBmRoohqiBYogaKIaogWKIGiiGqIFiiBoohqiBYogaKIaogWKIGiiG\nqIFiiBoohqiBYogaKIaogWKcpP2D2l9L+nvLh/2lpH+0fMwuTdPeadoqTdferrb+Ksm6y93RSdRd\nsL2YZL7vHU1N095p2ipN194+tvL0GyiGqIFipinqPX0PGNE07Z2mrdJ07Z341ql5TQ2gmWk6UwNo\ngKiBYqYiats7bX9q+5TtJ/recyW299o+Y/vjvrcMY3uD7YO2T9g+bnt335uWY3uN7fdtfzTY+lTf\nm5qwPWP7Q9v7JvWYKz5q2zOSnpV0l6RNknbZ3tTvqit6QdLOvkc0dEHSY0k2Sdom6fcr+N/2nKQd\nSX4jabOknba39bypid2STk7yAVd81JK2SjqV5LMk57X0lzfv73nTspK8I+mbvnc0keSrJEcG73+v\npS++9f2uurwsOTu4OTt4W9FXeW3PSbpH0nOTfNxpiHq9pC8uuX1aK/QLb5rZ3ihpi6RD/S5Z3uCp\n7FFJZyQdSLJitw48I+lxST9O8kGnIWp0zPb1kl6T9GiS7/res5wkF5NsljQnaavt2/retBzb90o6\nk+TwpB97GqL+UtKGS27PDT6GFtie1VLQLyV5ve89TST5VtJBrexrF9sl3Wf7cy29ZNxh+8VJPPA0\nRP2BpFts32z7Gi394fs3et5Ugm1Lel7SySRP973nSmyvs33D4P1rJd0h6ZN+Vy0vyZNJ5pJs1NLX\n7FtJHpzEY6/4qJNckPSIpP1aupDzapLj/a5anu2XJb0n6Vbbp20/3PemK9gu6SEtnUWODt7u7nvU\nMm6UdND2MS39R38gycS+TTRN+DFRoJgVf6YGMBqiBoohaqAYogaKIWqgGKIGiiFqoJj/AoxH0nIr\nac7DAAAAAElFTkSuQmCC\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "stream",
          "text": [
            "[0.75 0.  ]\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAPUAAAD4CAYAAAA0L6C7AAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjIsIGh0\ndHA6Ly9tYXRwbG90bGliLm9yZy8li6FKAAAIuElEQVR4nO3dzYtdhR3G8efpZMz4UhDaLGwmNC6s\nEKRNYEgD2aVI4wu6NaArYTYVIgiiS/+AWjduBg0WFEXQhQRLCDUigo1OYhSTKASxGBFiK6JpaWLi\n08VcSiqZ3HNvzrln7o/vBwbmzh3OfQjzzbn3zDDjJAJQx0/6HgCgXUQNFEPUQDFEDRRD1EAx67o4\n6DVenzld38WhAUj6j/6l8znny93XSdRzul6/9e+6ODQASYfz11Xv4+k3UAxRA8UQNVAMUQPFEDVQ\nDFEDxRA1UAxRA8UQNVAMUQPFEDVQDFEDxRA1UAxRA8UQNVAMUQPFEDVQTKOobe+2/YntU7Yf63oU\ngPENjdr2jKSnJd0haYukPba3dD0MwHianKm3SzqV5NMk5yW9JOnebmcBGFeTqDdK+vyS26cHH/s/\nthdtL9te/l7n2toHYEStXShLspRkIcnCrNa3dVgAI2oS9ReSNl1ye37wMQBrUJOo35N0i+2bbV8j\n6T5Jr3U7C8C4hv4y/yQXbD8k6YCkGUn7khzvfBmAsTT6Cx1JXpf0esdbALSAnygDiiFqoBiiBooh\naqAYogaKIWqgGKIGiiFqoBiiBoohaqAYogaKIWqgGKIGiiFqoBiiBoohaqAYogaKIWqgGKIGiiFq\noBiiBoohaqAYogaKIWqgGKIGiiFqoBiiBoohaqAYogaKIWqgGKIGiiFqoBiiBoohaqCYoVHb3mf7\njO2PJjEIwNVpcqZ+TtLujncAaMnQqJO8JenrCWwB0AJeUwPFrGvrQLYXJS1K0pyua+uwAEbU2pk6\nyVKShSQLs1rf1mEBjIin30AxTb6l9aKkdyTdavu07Qe7nwVgXENfUyfZM4khANrB02+gGKIGiiFq\noBiiBoohaqAYogaKIWqgGKIGiiFqoBiiBoohaqAYogaKIWqgGKIGiiFqoBiiBoohaqAYogaKIWqg\nGKIGiiFqoBiiBoohaqAYogaKIWqgGKIGiiFqoBiiBoohaqAYogaKIWqgGKIGiiFqoBiiBoohaqCY\noVHb3mT7kO0Tto/b3juJYQDGs67B51yQ9EiSo7Z/KumI7YNJTnS8DcAYhp6pk3yZ5Ojg/e8knZS0\nsethAMbT5Ez9P7Y3S9om6fBl7luUtChJc7quhWkAxtH4QpntGyS9IunhJN/++P4kS0kWkizMan2b\nGwGMoFHUtme1EvQLSV7tdhKAq9Hk6rclPSvpZJInu58E4Go0OVPvlPSApF22jw3e7ux4F4AxDb1Q\nluRtSZ7AFgAt4CfKgGKIGiiGqIFiiBoohqiBYogaKIaogWKIGiiGqIFiiBoohqiBYogaKIaogWKI\nGiiGqIFiiBoohqiBYogaKIaogWKIGiiGqIFiiBoohqiBYogaKIaogWKIGiiGqIFiiBoohqiBYoga\nKIaogWKIGiiGqIFiiBooZmjUtudsv2v7A9vHbT8xiWEAxrOuweeck7QryVnbs5Letv2XJH/reBuA\nMQyNOkkknR3cnB28pctRAMbX6DW17RnbxySdkXQwyeFuZwEYV6Ook1xMslXSvKTttm/78efYXrS9\nbHv5e51reyeAhka6+p3kG0mHJO2+zH1LSRaSLMxqfVv7AIyoydXvDbZvHLx/raTbJX3c9TAA42ly\n9fsmSX+2PaOV/wReTrK/21kAxtXk6veHkrZNYAuAFvATZUAxRA0UQ9RAMUQNFEPUQDFEDRRD1EAx\nRA0UQ9RAMUQNFEPUQDFEDRRD1EAxRA0UQ9RAMUQNFNPkN5+M7Fe//rcOHDjWxaFb9/tfbO17AtaI\nU3/a0feExs79cfVfu8+ZGiiGqIFiiBoohqiBYogaKIaogWKIGiiGqIFiiBoohqiBYogaKIaogWKI\nGiiGqIFiiBoohqiBYogaKIaogWIaR217xvb7tvd3OQjA1RnlTL1X0smuhgBoR6Oobc9LukvSM93O\nAXC1mp6pn5L0qKQfVvsE24u2l20vf/XPi62MAzC6oVHbvlvSmSRHrvR5SZaSLCRZ2PCzmdYGAhhN\nkzP1Tkn32P5M0kuSdtl+vtNVAMY2NOokjyeZT7JZ0n2S3khyf+fLAIyF71MDxYz0Z3eSvCnpzU6W\nAGgFZ2qgGKIGiiFqoBiiBoohaqAYogaKIWqgGKIGiiFqoBiiBoohaqAYogaKIWqgGKIGiiFqoBii\nBopxkvYPan8l6e8tH/bnkv7R8jG7NE17p2mrNF17u9r6yyQbLndHJ1F3wfZykoW+dzQ1TXunaas0\nXXv72MrTb6AYogaKmaaol/oeMKJp2jtNW6Xp2jvxrVPzmhpAM9N0pgbQAFEDxUxF1LZ32/7E9inb\nj/W950ps77N9xvZHfW8ZxvYm24dsn7B93Pbevjetxvac7XdtfzDY+kTfm5qwPWP7fdv7J/WYaz5q\n2zOSnpZ0h6QtkvbY3tLvqit6TtLuvkc0dEHSI0m2SNoh6Q9r+N/2nKRdSX4jaauk3bZ39Lypib2S\nTk7yAdd81JK2SzqV5NMk57Xylzfv7XnTqpK8Jenrvnc0keTLJEcH73+nlS++jf2uurysODu4OTt4\nW9NXeW3PS7pL0jOTfNxpiHqjpM8vuX1aa/QLb5rZ3ixpm6TD/S5Z3eCp7DFJZyQdTLJmtw48JelR\nST9M8kGnIWp0zPYNkl6R9HCSb/ves5okF5NslTQvabvt2/retBrbd0s6k+TIpB97GqL+QtKmS27P\nDz6GFtie1UrQLyR5te89TST5RtIhre1rFzsl3WP7M628ZNxl+/lJPPA0RP2epFts32z7Gq384fvX\net5Ugm1LelbSySRP9r3nSmxvsH3j4P1rJd0u6eN+V60uyeNJ5pNs1srX7BtJ7p/EY6/5qJNckPSQ\npANauZDzcpLj/a5ane0XJb0j6Vbbp20/2PemK9gp6QGtnEWODd7u7HvUKm6SdMj2h1r5j/5gkol9\nm2ia8GOiQDFr/kwNYDREDRRD1EAxRA0UQ9RAMUQNFEPUQDH/Ba/g0CnVW2MhAAAAAElFTkSuQmCC\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "stream",
          "text": [
            "[1. 0.]\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\nIWqgmGmK+kDfA0Y0TXunaas0XXsnvnVqXlMDaGaaztQAGiBqoJipiNr2Xtuf2T5r+4m+91yN7YO2\nl21/0veWYWxvsX3U9mnbp2zv73vTWmzP2X7f9keDrU/1vakJ2zO2P7R9aFKPue6jtj0j6VlJd0na\nJmmf7W39rrqqFyTt7XtEQ5ckPZZkm6Rdkv64jv9tVyTtSfJbSdsl7bW9q+dNTeyXdGaSD7juo5a0\nU9LZJJ8nuaDVv7x5f8+b1pTkHUnf9L2jiSRfJzkxeP97rX7xbe531ZVl1fnBzdnB27q+ymt7XtI9\nkp6b5ONOQ9SbJX152e1zWqdfeNPM9lZJOyQd63fJ2gZPZU9KWpZ0JMm63TrwjKTHJf00yQedhqjR\nMds3SXpN0qNJvut7z1qS/Jhku6R5STtt39H3prXYvlfScpLjk37saYj6K0lbLrs9P/gYWmB7VqtB\nv5Tk9b73NJHkW0lHtb6vXeyWdJ/tL7T6knGP7Rcn8cDTEPUHkm6zfavt67T6h+/f6HlTCbYt6XlJ\nZ5I83feeq7G9yfbNg/evl3SnpE/7XbW2JE8mmU+yVatfs28leXASj73uo05ySdIjkg5r9ULOq0lO\n9btqbbZflvSepNttn7P9cN+brmK3pIe0ehY5OXi7u+9Ra7hF0lHbH2v1P/ojSSb2baJpwo+JAsWs\n+zM1gNEQNVAMUQPFEDVQDFEDxRA1UAxRA8X8F2XczgyP3pVEAAAAAElFTkSuQmCC\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "stream",
          "text": [
            "[1.   0.25]\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "stream",
          "text": [
            "[1.  0.5]\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "stream",
          "text": [
            "[1.   0.75]\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\nvbO0VZqtvVPfOjOvqQE0M0tnagANEDVQzExEbXuP7c9tn7H9ZN97rsb2Adurtj/te8sotrfYPmL7\nlO2Ttvf1vWktthdsf2D74+HWp/ve1ITtOdsf2T44rcdc91HbnpP0nKS7JW2TtNf2tn5XXdWLkvb0\nPaKhS5IeT7JN0p2S/rCO/23PS9qd5DeStkvaY/vOnjc1sU/S6Wk+4LqPWtJOSWeSfJHkggZ/efOB\nnjetKcm7kr7te0cTSb5Jcnz4/g8afPFt7nfVlWXg3PDm/PBtXV/ltb0o6V5Jz0/zcWch6s2Svrrs\n9lmt0y+8WWZ7q6Qdko72u2Rtw6eyJyStSjqcZN1uHXpW0hOSfprmg85C1OiY7ZskvS7psSTf971n\nLUl+TLJd0qKknbbv6HvTWmzfJ2k1ybFpP/YsRP21pC2X3V4cfgwtsD2vQdAvJ3mj7z1NJPlO0hGt\n72sXuyTdb/tLDV4y7rb90jQeeBai/lDSbbZvtX2dBn/4/s2eN5Vg25JekHQ6yTN977ka25ts3zx8\n/3pJd0n6rN9Va0vyVJLFJFs1+Jp9O8lD03jsdR91kkuSHpV0SIMLOa8lOdnvqrXZfkXS+5Jut33W\n9iN9b7qKXZIe1uAscmL4dk/fo9Zwi6Qjtj/R4D/6w0mm9m2iWcKPiQLFrPszNYDxEDVQDFEDxRA1\nUAxRA8UQNVAMUQPF/Adfs8r9MZ+ccgAAAABJRU5ErkJggg==\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "jsDWXDtPdW7d",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "class QLearningAgent:\n",
        "    def __init__(self, env, epsilon=1.0, lr=0.1, gamma=0.9):\n",
        "        self.env = env\n",
        "        self.observation_space = env.observation_space\n",
        "        self.action_space = env.action_space\n",
        "        q_table_dim = env.observation_space.shape[0] + 1\n",
        "        self.q_table = np.zeros((q_table_dim, q_table_dim, env.action_space.n))\n",
        "        self.epsilon = epsilon\n",
        "        self.lr = lr\n",
        "        self.gamma = gamma\n",
        "\n",
        "    def policy(self, observation):\n",
        "      # Code for policy (Task 1) (30 points)\n",
        "      if np.random.random() < self.epsilon:\n",
        "        action = np.random.choice(self.action_space.n)\n",
        "      else:\n",
        "        observation = observation.astype(int)\n",
        "        action = np.argmax(self.q_table[observation[0]][observation[1]])\n",
        "      return(action)\n",
        "        \n",
        "    def step(self, observation):\n",
        "      return self.policy(observation)\n",
        "        \n",
        "    def update(self, state, action, reward, next_state):\n",
        "        state = state.astype(int)\n",
        "        next_state = next_state.astype(int)\n",
        "        # Code for updating Q Table (Task 2) (20 points)\n",
        "        self.q_table[state[0]][state[1]][action] = (1- self.lr)*self.q_table[state[0]][state[1]][action] + self.lr*(reward + self.gamma*(np.max(self.q_table[next_state[0]][next_state[1]][action])))\n",
        "        return self.q_table\n",
        "        \n",
        "    def set_epsilon(self, epsilon):\n",
        "        self.epsilon = epsilon"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "TNpaUXYJmqce",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "env = GridEnvironment() # note: we do not normalize\n",
        "agent = QLearningAgent(env)\n",
        "episodes = 1000 # number of games we want the agent to play\n",
        "delta_epsilon = agent.epsilon/episodes\n",
        "decay_rate = 0.99\n",
        "\n",
        "total_rewards = []\n",
        "epsilons = []\n",
        "\n",
        "# Training Process (Task 3) (20 points)\n",
        "for i in range(episodes):\n",
        "  obs = env.reset()\n",
        "  done = False\n",
        "  epsilons.append(agent.epsilon)\n",
        "  while (done == False):\n",
        "    action = agent.step(obs)\n",
        "    #current_state = obs\n",
        "    new_state, reward, done, info = env.step(action)\n",
        "    total_rewards.append(reward)\n",
        "    agent.update(obs, action, reward, new_state)\n",
        "    obs = new_state.copy()\n",
        "  agent.set_epsilon(max(agent.epsilon*decay_rate, .1))   \n",
        "\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Ob4gmxGCzvu_",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 296
        },
        "outputId": "4749ad4d-cb0d-4139-ad9a-9bf0e8e01f52"
      },
      "source": [
        "plt.xlabel('Episode')\n",
        "plt.ylabel('$\\epsilon$')\n",
        "plt.plot(epsilons)"
      ],
      "execution_count": 49,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[<matplotlib.lines.Line2D at 0x7f6120a04cc0>]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 49
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LXbxcuADzyhY",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 296
        },
        "outputId": "7ce94c32-4e26-4760-905f-379aa0e560d8"
      },
      "source": [
        "window = 10\n",
        "plt.xlabel('Episode')\n",
        "plt.ylabel('Total Reward (SMA 10)')\n",
        "plt.plot([np.mean(total_rewards[tr:tr+window]) for tr in range(window, len(total_rewards))])"
      ],
      "execution_count": 50,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[<matplotlib.lines.Line2D at 0x7f612039c358>]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 50
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\nGx8AAAAASUVORK5CYII=\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "9j0jGpPDz0yU",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "outputId": "52c79a51-3517-48a8-9a8d-c04a61c441cd"
      },
      "source": [
        "env = GridEnvironment()\n",
        "\n",
        "obs = env.reset()\n",
        "done = False\n",
        "agent.epsilon = 0\n",
        "env.render()\n",
        "plt.show()\n",
        "\n",
        "while not done:\n",
        "    action = agent.step(obs)\n",
        "    obs, reward, done, info = env.step(action)\n",
        "    env.render()\n",
        "    plt.show()"
      ],
      "execution_count": 51,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\nAAAAAElFTkSuQmCC\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\njLoY2OEAAAAASUVORK5CYII=\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n1F2wvZBkvu8dTU3T3mnaKk3X3j62cvcbKIaogWKmKeo9fQ8Y0TTtnaat0nTtnfjWqXlMDaCZaTpT\nA2iAqIFipiJq2ztsf2b7lO0n+95zJbb32j5j+5O+twxje73tg7ZP2D5ue3ffm5Zje43tD2x/PNj6\ndN+bmrA9Y/sj2/smdZsrPmrbM5Kek3S3pI2Sdtre2O+qK3pR0o6+RzR0QdLjSTZK2irp9yv433ZR\n0vYkv5G0SdIO21t73tTEbkknJ3mDKz5qSVsknUryeZJzWvrLmw/0vGlZSd6V9G3fO5pI8nWSI4PP\nf9DSD9+6flddXpacHVycHXys6Gd5bc9JulfS85O83WmIep2kLy+5fFor9AdvmtneIGmzpEP9Llne\n4K7sUUlnJB1IsmK3Djwr6QlJP07yRqchanTM9g2SXpf0WJLv+96znCQXk2ySNCdpi+3b+960HNv3\nSTqT5PCkb3saov5K0vpLLs8NvoYW2J7VUtAvJ3mj7z1NJPlO0kGt7Ocutkm63/YXWnrIuN32S5O4\n4WmI+kNJt9q+xfYqLf3h+zd73lSCbUt6QdLJJM/0vedKbK+1fePg82sl3Snp035XLS/JU0nmkmzQ\n0s/s20kemsRtr/iok1yQ9Kik/Vp6Iue1JMf7XbU8269Iel/SbbZP236k701XsE3Sw1o6ixwdfNzT\n96hl3CTpoO1jWvqP/kCSib1MNE34NVGgmBV/pgYwGqIGiiFqoBiiBoohaqAYogaKIWqgmP8C+HDV\njKyKl5oAAAAASUVORK5CYII=\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\ntHeatkrTtbePrTz9BoohaqCYaYp6qe8BI5qmvdO0VZquvRPfOjXfUwNoZprO1AAaIGqgmKmI2vaC\n7c9tn7b9ZN97rsT2fttnbX/a95ZhbG+xfdj2SdsnbO/re9NabG+0/YHtjwdbn+57UxO2Z2x/ZPvA\npB5z3Udte0bSc5LulrRN0l7b2/pddUUvSlroe0RDFyU9nmSbpF2S/riO/21XJO1J8jtJ2yUt2N7V\n86Ym9kk6NckHXPdRS9op6XSSL5Kc1+pf3nyg501rSvKupG/73tFEkm+SHBu8/4NWv/g297vq8rLq\n3ODm7OBtXV/ltT0n6V5Jz0/ycach6s2Svrrk9hmt0y+8aWZ7q6Qdko70u2Rtg6eyxyWdlXQoybrd\nOvCspCck/TzJB52GqNEx2zdKel3SY0m+73vPWpL8lGS7pDlJO23f0femtdi+T9LZJEcn/djTEPXX\nkrZccntu8DG0wPasVoN+Ockbfe9pIsl3kg5rfV+72C3pfttfavVbxj22X5rEA09D1B9Kus32rbav\n1eofvn+z500l2LakFySdSvJM33uuxPYm2zcN3r9O0p2SPut31dqSPJVkLslWrX7Nvp3koUk89rqP\nOslFSY9KOqjVCzmvJTnR76q12X5F0vuSbrd9xvYjfW+6gt2SHtbqWeT44O2evket4WZJh21/otX/\n6A8lmdiPiaYJvyYKFLPuz9QARkPUQDFEDRRD1EAxRA0UQ9RAMUQNFPNfKS/OoQi9o6cAAAAASUVO\nRK5CYII=\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\nYII=\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\nQmCC\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\nYII=\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\nvbO0VZqtvVPfOjOvqQE0M0tnagANEDVQzExEbXuP7c9tn7H9ZN97rsb2Adurtj/te8sotrfYPmL7\nlO2Ttvf1vWktthdsf2D74+HWp/ve1ITtOdsf2T44rcdc91HbnpP0nKS7JW2TtNf2tn5XXdWLkvb0\nPaKhS5IeT7JN0p2S/rCO/23PS9qd5DeStkvaY/vOnjc1sU/S6Wk+4LqPWtJOSWeSfJHkggZ/efOB\nnjetKcm7kr7te0cTSb5Jcnz4/g8afPFt7nfVlWXg3PDm/PBtXV/ltb0o6V5Jz0/zcWch6s2Svrrs\n9lmt0y+8WWZ7q6Qdko72u2Rtw6eyJyStSjqcZN1uHXpW0hOSfprmg85C1OiY7ZskvS7psSTf971n\nLUl+TLJd0qKknbbv6HvTWmzfJ2k1ybFpP/YsRP21pC2X3V4cfgwtsD2vQdAvJ3mj7z1NJPlO0hGt\n72sXuyTdb/tLDV4y7rb90jQeeBai/lDSbbZvtX2dBn/4/s2eN5Vg25JekHQ6yTN977ka25ts3zx8\n/3pJd0n6rN9Va0vyVJLFJFs1+Jp9O8lD03jsdR91kkuSHpV0SIMLOa8lOdnvqrXZfkXS+5Jut33W\n9iN9b7qKXZIe1uAscmL4dk/fo9Zwi6Qjtj/R4D/6w0mm9m2iWcKPiQLFrPszNYDxEDVQDFEDxRA1\nUAxRA8UQNVAMUQPF/Adfs8r9MZ+ccgAAAABJRU5ErkJggg==\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "5vYub8l6_nDk",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 510
        },
        "outputId": "2fd4eab8-b583-41a5-812e-90d66b5ef1f0"
      },
      "source": [
        "agent.q_table"
      ],
      "execution_count": 55,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[[ 3.28406185, -4.52843358,  1.33457273, -4.12963218],\n",
              "        [ 2.77361404, -0.77255306,  0.49951   , -1.23164736],\n",
              "        [ 1.01742328, -0.29701   ,  0.1       , -0.32104842],\n",
              "        [ 0.19      , -0.1       ,  0.1       ,  0.        ],\n",
              "        [ 0.        ,  0.        ,  0.        ,  0.        ]],\n",
              "\n",
              "       [[ 2.53784651, -4.81102876,  2.17791573, -4.69094457],\n",
              "        [ 2.31025177, -1.31886212,  1.3392059 , -1.68868358],\n",
              "        [ 1.30276671, -0.62952666,  0.558559  , -0.86241479],\n",
              "        [ 1.00623331, -0.2071    ,  0.1       , -0.29030128],\n",
              "        [ 0.19      ,  0.        , -0.1       , -0.11126839]],\n",
              "\n",
              "       [[ 1.70871834, -4.91460124,  2.13821819, -3.17445405],\n",
              "        [ 1.45509249, -1.28054974,  1.27491666, -2.17887298],\n",
              "        [ 0.4177    , -0.66027353,  1.29100824, -0.61575919],\n",
              "        [ 1.19576637, -0.53845123,  0.3439    , -0.40257124],\n",
              "        [ 0.83709372, -0.19      ,  0.        ,  0.        ]],\n",
              "\n",
              "       [[ 0.78746483, -4.16139164,  2.9037719 , -2.60299627],\n",
              "        [ 0.50492544, -1.91000668,  1.98251719, -2.60877022],\n",
              "        [ 0.47229964, -1.30362724,  1.07591533, -2.63213613],\n",
              "        [ 0.07268289, -1.33339754,  0.07143461, -2.48974643],\n",
              "        [ 1.        , -0.54868259, -1.04661746, -1.76910947]],\n",
              "\n",
              "       [[-0.199     , -0.42007699,  2.84996944, -0.199     ],\n",
              "        [-0.58519851, -0.48279273,  2.6757    , -0.383761  ],\n",
              "        [-0.58519851, -0.59934316,  1.89916662, -0.60998747],\n",
              "        [-1.04661746, -1.16741484,  1.        , -1.08835707],\n",
              "        [ 0.        ,  0.        ,  0.        ,  0.        ]]])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 55
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_YzFK-9v_pvE",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "d0c878fc-29cf-4d0a-e6de-0089c5c5b0a1"
      },
      "source": [
        ""
      ],
      "execution_count": 74,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "4\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "dyRZCf0MCkBD",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "0429240d-a5b2-45f7-e8a3-a330493b7638"
      },
      "source": [
        ""
      ],
      "execution_count": 67,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "2\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "oDYJ6fPhIAKo",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}