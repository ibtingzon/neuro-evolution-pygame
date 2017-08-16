from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import json
import numpy as np
import os

class Agent():
    def __init__(self, env):
        self.env = env
        self.actions = self.env.getActionSet()
        self.model = self.initModel()
        self.fitness = 0

    def initModel(self):
      model = Sequential()
      model.add(Dense(output_dim=7, input_dim=2, activation='relu'))
      model.add(Dense(output_dim=3, activation='softmax'))
      model.add(Activation("sigmoid"))
      model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      return model

    def predict(self, dist, ball_x):
      input_ = [dist/self.env.game.height, ball_x/self.env.game.width]
      neural_input = np.asarray(input_)
      neural_input = np.atleast_2d(neural_input)
      pred = self.model.predict_classes(neural_input, verbose=0)[0]
      return pred

    def setFitness(self, reward):
      self.fitness = reward

    def saveModel(self, gen):
      path = "models/gen" + str(gen) 
      if not os.path.exists(path):
        os.makedirs(path)
      self.model.save_weights(path + '/' + str(self.id) + ".keras")

    def loadModel(self, gen, id_):
      self.id = id_
      path = "models/gen" + str(gen) 
      self.model.load_weights(path + '/' + str(id_) + ".keras")

    def chooseAction(self, state):
      return self.actions[self.predict(state['player_y'] - state['ball_y'], state['ball_x'])]