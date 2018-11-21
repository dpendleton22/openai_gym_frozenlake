"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.s = 0
        self.a = 0

        """
        @Q Table Steps
        1. Init Q[]
        2. Compute the State (S)
        3. Select an Action (a)
        4. Observe (s_prime, reward)
        5. Update the Q-table
        """
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr

        self.dyna = dyna
        self.dyna_events = []

        self.q_Table = np.zeros(shape=(num_states, num_actions), dtype=float)

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = rand.randint(0, self.num_actions-1)
        self.a = action
        if self.verbose:
            print ('START')
            print ("s =", s,"a =",action)
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The new state
        @returns: The selected action
        """
        prev_state = self.s
        prev_action = self.a
        self.q_Table[prev_state, prev_action] = ((1-self.alpha)*(self.q_Table[prev_state, prev_action])) + (self.alpha * (r + (self.gamma * self.q_Table[s_prime, np.argmax(self.q_Table[s_prime])])))

        random_act = rand.random()
        if random_act < self.rar:
            random_action = rand.randint(0, self.num_actions-1)
            action = random_action
        else:
            action = np.argmax(self.q_Table[s_prime])

        self.dyna_events.append((prev_state, prev_action, s_prime, r))
        dyna_len = len(self.dyna_events)
        i = 0
        while i < self.dyna:
            hallucenate_event = self.dyna_events[rand.randint(0, dyna_len-1)]
            dyna_state = hallucenate_event[0]
            dyna_action = hallucenate_event[1]
            dyna_s_prime = hallucenate_event[2]
            dyna_r = hallucenate_event[3]
            self.q_Table[dyna_state, dyna_action] = ((1 - self.alpha) * (self.q_Table[dyna_state, dyna_action])) + (
                        self.alpha * (dyna_r + (self.gamma * self.q_Table[dyna_s_prime, np.argmax(self.q_Table[dyna_s_prime])])))
            i+=1
        self.a = action
        self.s = s_prime
        self.rar = self.rar * self.radr
        if self.verbose: print ("s =", s_prime,"a =",action,"r =",r)
        return action

    def author(self):
        return 'dpendleton6'

if __name__=="__main__":
    print ("Remember Q from Star Trek? Well, this isn't him")
