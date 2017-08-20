import numpy as np
import gym
import numpy as np
import genetic
import random
import gc
from agent import Agent

MAX_GEN = 100
SIZE = 20

def endGen(gen, players):
    elite_players = []
    while len(elite_players) < SIZE:
        best_player = genetic.getBest(list(set(players) - set(elite_players)))
        elite_players.append(best_player)

    print("\n\nEvolving new generation...")
    new_weights = []
    while len(new_weights) < SIZE:
        if len(new_weights) % 5 == 0:
            print(str(len(new_weights)) + ' of ' + str(SIZE)  + ' done...')

        parents = genetic.tournament(elite_players[:int(SIZE/2)], 2, 5)
        parent1 = parents[0].model.get_weights()
        parent2 = parents[1].model.get_weights()

        child1, child2 = genetic.crossover(parent1, parent2)
        chosen = random.choice([child1, child2])
        chosen = genetic.mutate(chosen)
        new_weights.append(chosen)

    print(str(len(new_weights)) + ' of ' + str(SIZE)  + ' done...')

    for player in elite_players[:int(SIZE/2)]:
        new_weights.append(player.model.get_weights())

    population = players[:int(SIZE)] + elite_players[:int(SIZE/2)]
    for player, weights in zip(population, new_weights):
        player.model.set_weights(weights)

    index = 0
    #if gen % 2 == 0:
    print("Saving new models")
    for index, player in enumerate(population):
        player.id = index
        player.saveModel(gen)

    gc.collect()

    return population

def main():
  gen = 0
  env = gym.make('CartPole-v0')
  env.reset()

  population = []
  if gen == 0:
    for i in range(SIZE):
      population.append(Agent(env))
  else:
    for i in range(SIZE + int(SIZE/2)):
      agent = Agent(env)
      agent.loadModel(gen, i)
      population.append(agent)

  while gen < MAX_GEN:
    print("Generation: ", gen)
    for i, agent in enumerate(population):
      observation = env.reset()
      total_reward = 0
      for t in range(10000):
        env.render()
        action = agent.chooseAction(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
          agent.setFitness(total_reward)
          print("Agent ", str(i), " Fitness: ", agent.fitness)
          print("Episode finished after {} timesteps".format(t+1))
          break
    gen += 1
    population = endGen(gen, population)

if __name__ == '__main__':
  main()