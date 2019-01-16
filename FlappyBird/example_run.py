
from __future__ import print_function
import tensorflow as tf
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np



ACTIONS = 2
def run():
    # open up a game state to communicate with emulator
    game_state = game.GameState()
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
   


    epsilon = INITIAL_EPSILON
    t = 0
    while "flappy bird" != "angry bird":
      
        a_t =[1,0] 
        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
      

       


def main():
    run()

if __name__ == "__main__":
    main()
