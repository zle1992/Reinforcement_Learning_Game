
from __future__ import print_function
import tensorflow as tf
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np



n_actions = 2
def run():
    # open up a game state to communicate with emulator
    game_state = game.GameState()
    do_nothing = np.zeros(n_actions)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
   


    t = 0
    while "flappy bird" != "angry bird":
      
        a_t = np.zeros(n_actions)
        action = np.random.randint(0,n_actions)
        print(action)
        a_t[action] =1 
        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
      

       


def main():
    run()

if __name__ == "__main__":
    main()
