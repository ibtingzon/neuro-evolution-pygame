import numpy as np
from ple import PLE
from ple.games.pong import Pong
from agent import Agent
import numpy as np
import genetic
import random
import gc

MAX_GEN = 100
SIZE = 15

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

        parents = genetic.tournament(elite_players[:int(SIZE/2)], 2, 3)
        parent1 = parents[0].model.get_weights()
        parent2 = parents[1].model.get_weights()

        child1, child2 = genetic.crossover(parent1, parent2)
        chosen = random.choice([child1, child2])
        if random.random() < 0.3:
          chosen = genetic.mutate(chosen)
        new_weights.append(chosen)

    print(str(len(new_weights)) + ' of ' + str(SIZE)  + ' done...')

    for player in elite_players[:int(SIZE/2)]:
        new_weights.append(player.model.get_weights())

    population = players[:int(SIZE)] + elite_players[:int(SIZE/2)]
    for player, weights in zip(population, new_weights):
        player.model.set_weights(weights)

    index = 0
    print("Saving new models")
    for index, player in enumerate(population):
      player.id = index
      player.saveModel(gen)

    gc.collect()

    return population

def main():
  gen = 2
  width = 500
  height = 300
  game = Pong(width, height, MAX_SCORE=3)
  env = PLE(game, fps=20, display_screen=True)

  population = []
  env.init()

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
      total_reward, time = 0, 0
      while True:
        state = env.game.getGameState()
        observation = env.getScreenRGB()
        action = agent.chooseAction(state)
        reward = env.act(action)
        total_reward += reward
        time += env.game.tick(env.fps)

        if reward != 0:
          curr_agent_score = env.game.score_counts['agent']
          curr_cpu_score = env.game.score_counts['cpu']
          print(".::Score Board::. Agent: ", curr_agent_score, " CPU: ", curr_cpu_score)

        if env.game_over():
          agent.setFitness(5*total_reward + 10*env.score() + time/1000)
          print("Agent ", i, " fitness: ", agent.fitness)
          break
      env.reset_game()
    population = endGen(gen, population)
    gen += 1

if __name__ == '__main__':
  main()