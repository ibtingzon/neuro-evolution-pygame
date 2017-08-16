import numpy as np
from ple import PLE
from ple.games.catcher import Catcher
from agent import Agent
import numpy as np
import genetic
import random
import gc

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
  gen = 6
  width = 300
  height = 500
  game = Catcher(300, 500)
  env = PLE(game, fps=120, display_screen=True)

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
      reward = 0

      curr_lives = env.game.lives
      curr_score = env.score()
      print("  Lives left:", curr_lives)
      print("  Score: ", curr_score,)

      while True:
        if env.game.lives < curr_lives:
          curr_lives = env.game.lives
          print("  Lives left:", curr_lives)

        if env.score() > curr_score:
          curr_score = env.score()
          print("  Score: ", curr_score)

        state = env.game.getGameState()
        observation = env.getScreenRGB()
        action = agent.chooseAction(state)
        reward += env.act(action)

        if env.game_over() or curr_score > 20:
          agent.setFitness(reward)
          print("Agent ", i, " fitness: ", agent.fitness)
          break
      env.reset_game()
    gen += 1
    population = endGen(gen, population)

if __name__ == '__main__':
  main()