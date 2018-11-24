'''
Frozen Lake MDP

MDP is defined by:
1. Set of all possible states: S = {so, s1,..., sn}
2. Initial State: s0
3. Set of all possible actions: A = {a0, a1,...,an}
4. Transition Model: T(s, a, s')
5. Reward Function: R(s)

Steps to take:
1. Implement value iteration
2. Create Policy evaluation
3. Implement policy iteration
'''
import numpy as np
import gym
import random
import QLearner as ql
from gym.envs.toy_text import frozen_lake

state_map = {'S': 0, 'F': 0, 'H': -1, 'G':10}

frozen_lake.MAPS['16x16'] = [
        "SFFFFFFFHFFFFFFF",
        "FFFFFFFFFFFFFFHH",
        "FFFHFFFFFHHFFFFH",
        "FFFFFHFFFFFFFFFF",
        "FFFHFFFFFFFFFFFF",
        "FHHFFFHFFFFFFFFF",
        "FHFFHFHFFFFFFFFF",
        "FHFFHFHFFFFFFFFF",
        "FHFFHFHFFFFFFFFF",
        "FHFFHFHFFFFHFFFF",
        "FHFFHFHFFFFHFFFF",
        "FHFFHFHFFFFFFFFF",
        "FHFFHFHFFFFFFFFF",
        "FHFFHFHFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFHHFFFG"
    ]

#FrozenLake-16x16-v0 is already registered locally
try:
    gym.envs.register(
        id='FrozenLake-16x16-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '16x16'},
        max_episode_steps=1000,
        reward_threshold=0.78, # optimum = .8196
    )
except:
    pass

'''
Class Value Iteration

@:param evn : pass in the Frozen Lake to implement value iteration 
'''
class value_iteration_mdp():
    def __init__(self, environment):
        self.env = environment
        self.nS = self.env.env.nS
        self.nA = self.env.env.nA
        self.state_list = np.arange(0, self.nS)
        self.state_grid = np.reshape(self.state_list, (int(np.sqrt(self.env.observation_space.n)),
                                     int(np.sqrt(self.env.observation_space.n))))

    def get_transition(self, row, col, action, tot_row, tot_col):
        """
        get tranisition takes the agents position and action passed it to determine
        the probability of moving into the next state with the given action
        """

        '''
        Expand the grid of the environment to handle when the 
        agent decides to move in the direction of a wall 
        '''
        state_probabilities = np.zeros((int(np.sqrt(self.env.observation_space.n)) + 2, int(np.sqrt(self.env.observation_space.n)) + 2), dtype=float)

        if action == 'UP':
            row += 1
            col += 1
            state_probabilities[row - 1, col] = 0.33 #UP
            state_probabilities[row, col - 1 ] = 0.33 #LEFT
            state_probabilities[row, col + 1] = 0.33 # RIGHT
            state_probabilities[row + 1, col] = 0.0 #DOWN
        elif action == 'LEFT':
            row += 1
            col += 1
            state_probabilities[row - 1, col] = 0.33 #UP
            state_probabilities[row, col - 1 ] = 0.33 #LEFT
            state_probabilities[row, col + 1] = 0.0 # RIGHT
            state_probabilities[row + 1, col] = 0.33 #DOWN
        elif action == 'RIGHT':
            row += 1
            col += 1
            state_probabilities[row - 1, col] = 0.33 #UP
            state_probabilities[row, col - 1 ] = 0.0 #LEFT
            state_probabilities[row, col + 1] = 0.33 # RIGHT
            state_probabilities[row + 1, col] = 0.33 #DOWN
        elif action == 'DOWN':
            row += 1
            col += 1
            state_probabilities[row - 1, col] = 0.0  # UP
            state_probabilities[row, col - 1] = 0.33  # LEFT
            state_probabilities[row, col + 1] = 0.33  # RIGHT
            state_probabilities[row + 1, col] = 0.33  # DOWN

        for row in range (0, tot_row+1):
            if state_probabilities[row, 0] != 0:
                state_probabilities[row, 1] += state_probabilities[row, 0]
            elif state_probabilities[row, -1] != 0:
                state_probabilities[row, -2] += state_probabilities[row, -1]

        for col in range (0, tot_col+1):
            if state_probabilities[0, col] != 0:
                state_probabilities[1, col] += state_probabilities[0, col]
            elif state_probabilities[-1, col] != 0:
                state_probabilities[-2, col] += state_probabilities[-1, col]

        return state_probabilities[1: 1+tot_row, 1:1+tot_col]


    def frozen_transition(self):
        T = np.zeros((self.nS, self.nS, self.nA), dtype=float)

        row_col = int(np.sqrt(self.nS))

        counter = 0
        for row in range(0, row_col):
            for col in range(0, row_col):
                line = self.get_transition(row, col, action="LEFT", tot_row=row_col, tot_col=row_col)
                T[counter, : , frozen_lake.LEFT] = line.flatten()
                line = self.get_transition(row, col, action="DOWN", tot_row=row_col, tot_col=row_col)
                T[counter, : , frozen_lake.DOWN] = line.flatten()
                line = self.get_transition(row, col, action="RIGHT", tot_row=row_col, tot_col=row_col)
                T[counter, : , frozen_lake.RIGHT] = line.flatten()
                line = self.get_transition(row, col, action="UP", tot_row=row_col, tot_col=row_col)
                T[counter, : , frozen_lake.UP] = line.flatten()

                counter += 1

        print("Done!")

        return T

    def return_state_utility(self, v, T, u, reward, gamma):
        """Return the state utility.

        @param v the value vector
        @param T transition matrix
        @param u utility vector
        @param reward for that state
        @param gamma discount factor
        @return the utility of the state
        """
        action_array = np.zeros(4)
        for action in range(0, 4):
            action_array[action] = np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
        return reward + gamma * np.max(action_array)

    def value_iteration(self):
        """
        Value iteration is a Markov Decision Process to iterate over a policy to find the utility function
        for each state in the environment. In MDP, the environment is known and that environments rewards are also known

        The value iteration is complete after the utility of each state is no longer updating

        :return:
        policy: array
        value: array
        """
        #Create a utility function of the environment shape
        gamma = 0.9
        epsilon = 0.01
        iteration = 0

        #create a utility function that matches the size of the number of states
        u = np.zeros(self.env.observation_space.n, dtype=float)

        u_copy = u.copy()

        #Create the reward grid
        reward = np.array([state_map.get(sublist) for state in frozen_lake.MAPS[self.env.spec._kwargs.get('map_name')] for sublist in state])

        T = self.frozen_transition()

        graph_list = list()

        #keep track of the convergence
        policy_convergence = list()

        while True:
            delta = 0
            iteration += 1
            u = u_copy.copy()
            graph_list.append(u)
            for s in range(self.env.observation_space.n):
                r = reward[s]
                v = np.zeros((1, self.env.observation_space.n), dtype=float)
                v[0, s] = 1.0
                u_copy[s] = self.return_state_utility(v, T, u, r, gamma)
                delta = max(delta, np.abs(u_copy[s] - u[s]))
                policy_convergence.append({'iter': iteration, 'delta': delta})
            if delta < epsilon * (1 - gamma) / gamma:
                print("Total Iterations: {}".format(iteration))
                print("=================== VALUE ITERATION RESULT ==================")
                print("Iterations: " + str(iteration))
                print("Delta: " + str(delta))
                print("Gamma: " + str(gamma))
                print("Epsilon: " + str(epsilon))
                print("===================================================")
                utility_reshape = np.reshape(u, (int(np.sqrt(self.env.observation_space.n)), int(np.sqrt(self.env.observation_space.n))))
                print (np.array(utility_reshape, dtype=float))
                print("===================================================")
                break

        return u

    def get_action_from_state(self, state, grid_policy):
        start_row = 1
        start_col = 1

        # determine the row of the state
        row = np.where(self.state_grid == state)[0][0]
        col = np.where(self.state_grid == state)[1][0]

        actions = {}

        action_key = {
            'LEFT': 0,
            'DOWN': 1,
            'RIGHT': 2,
            'UP': 3
        }

        row+= 1
        col+=1

        for x in range(4):
            if x == 0:
                actions['LEFT'] = grid_policy[row, col - 1]
            elif x == 1:
                actions['DOWN'] = grid_policy[row + 1, col]
            elif x == 2:
                actions['RIGHT'] = grid_policy[row, col + 1]
            elif x == 3:
                actions['UP'] = grid_policy[row - 1, col]

        action = max(actions, key=actions.get)

        get_action = action_key.get(action)

        return get_action

    def test_policy(self, policy, value=True):
        grid_policy = np.full((int(np.sqrt(self.env.observation_space.n))  + 2, int(np.sqrt(self.env.observation_space.n))  + 2), -999.0)
        value_row_start = 0
        value_row_end = int(np.sqrt(self.env.observation_space.n))
        for row in range(int(np.sqrt(self.env.observation_space.n)) ):
            row += 1
            grid_policy[row, 1:-1] = policy[value_row_start:value_row_end]
            value_row_start = value_row_end
            value_row_end += int(np.sqrt(self.env.observation_space.n))

        tot_reward = 0
        for i in range(100):
            print('Run: {}'.format(i))
            observation = self.env.reset()
            done = False
            if value:
                action = self.get_action_from_state(0, grid_policy)
            else:
                action = policy[observation]
            while not done:
                # self.env.render()
                observation, reward, done, info = self.env.step(int(action))
                # self.env.render()
                if value:
                    action = self.get_action_from_state(0, grid_policy)
                else:
                    action = policy[observation]
                tot_reward += reward
        print('Total Reward: {}'.format(tot_reward))
        return tot_reward

class policy_iteration():
    def __init__(self, environment):
        self.env = environment
        self.nS = self.env.env.nS
        self.nA = self.env.env.nA
        self.state_list = np.arange(0, self.nS)
        self.state_grid = np.reshape(self.state_list, (int(np.sqrt(self.env.observation_space.n)),
                                     int(np.sqrt(self.env.observation_space.n))))

    def return_policy_evaluation(self, p, u, r, T, gamma):
        """Return the policy utility.

        @param p policy vector
        @param u utility vector
        @param r reward vector
        @param T transition matrix
        @param gamma discount factor
        @return the utility vector u
        """
        for s in range(0, self.env.observation_space.n):
            if not np.isnan(p[s]):
                v = np.zeros((1, self.env.observation_space.n), dtype=float)
                v[0, s] = 1.0
                action = int(p[s])
                u[s] = r[s] + gamma * np.sum(np.multiply(u, np.dot(v, T[:, :, action])))
        return u

    def return_expected_action(self, u, T, v):
        """Return the expected action.

        It returns an action based on the
        expected utility of doing a in state s,
        according to T and u. This action is
        the one that maximize the expected
        utility.
        @param u utility vector
        @param T transition matrix
        @param v starting vector
        @return expected action (int)
        """
        actions_array = np.zeros(4)
        for action in range(4):
           #Expected utility of doing a in state s, according to T and u.
           actions_array[action] = np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
        return np.argmax(actions_array)

    def frozen_transition(self):
        T = np.zeros((self.nS, self.nS, self.nA), dtype=float)

        row_col = int(np.sqrt(self.nS))

        counter = 0
        for row in range(0, row_col):
            for col in range(0, row_col):
                line = self.get_transition(row, col, action="LEFT", tot_row=row_col, tot_col=row_col)
                T[counter, : , frozen_lake.LEFT] = line.flatten()
                line = self.get_transition(row, col, action="DOWN", tot_row=row_col, tot_col=row_col)
                T[counter, : , frozen_lake.DOWN] = line.flatten()
                line = self.get_transition(row, col, action="RIGHT", tot_row=row_col, tot_col=row_col)
                T[counter, : , frozen_lake.RIGHT] = line.flatten()
                line = self.get_transition(row, col, action="UP", tot_row=row_col, tot_col=row_col)
                T[counter, : , frozen_lake.UP] = line.flatten()

                counter += 1

        print("Done!")

        return T

    def get_transition(self, row, col, action, tot_row, tot_col):
        """
        get tranisition takes the agents position and action passed it to determine
        the probability of moving into the next state with the given action
        """

        '''
        Expand the grid of the environment to handle when the 
        agent decides to move in the direction of a wall 
        '''
        state_probabilities = np.zeros((int(np.sqrt(self.env.observation_space.n)) + 2, int(np.sqrt(self.env.observation_space.n)) + 2), dtype=float)

        if action == 'UP':
            row += 1
            col += 1
            state_probabilities[row - 1, col] = 0.33 #UP
            state_probabilities[row, col - 1 ] = 0.33 #LEFT
            state_probabilities[row, col + 1] = 0.33 # RIGHT
            state_probabilities[row + 1, col] = 0.0 #DOWN
        elif action == 'LEFT':
            row += 1
            col += 1
            state_probabilities[row - 1, col] = 0.33 #UP
            state_probabilities[row, col - 1 ] = 0.33 #LEFT
            state_probabilities[row, col + 1] = 0.0 # RIGHT
            state_probabilities[row + 1, col] = 0.33 #DOWN
        elif action == 'RIGHT':
            row += 1
            col += 1
            state_probabilities[row - 1, col] = 0.33 #UP
            state_probabilities[row, col - 1 ] = 0.0 #LEFT
            state_probabilities[row, col + 1] = 0.33 # RIGHT
            state_probabilities[row + 1, col] = 0.33 #DOWN
        elif action == 'DOWN':
            row += 1
            col += 1
            state_probabilities[row - 1, col] = 0.0  # UP
            state_probabilities[row, col - 1] = 0.33  # LEFT
            state_probabilities[row, col + 1] = 0.33  # RIGHT
            state_probabilities[row + 1, col] = 0.33  # DOWN

        for row in range (0, tot_row+1):
            if state_probabilities[row, 0] != 0:
                state_probabilities[row, 1] += state_probabilities[row, 0]
            elif state_probabilities[row, -1] != 0:
                state_probabilities[row, -2] += state_probabilities[row, -1]

        for col in range (0, tot_col+1):
            if state_probabilities[0, col] != 0:
                state_probabilities[1, col] += state_probabilities[0, col]
            elif state_probabilities[-1, col] != 0:
                state_probabilities[-2, col] += state_probabilities[-1, col]

        return state_probabilities[1: 1+tot_row, 1:1+tot_col]

    def print_policy(self, p, shape):
        """Printing utility.

        Print the policy actions using symbols:
        ^, v, <, > up, down, left, right
        * terminal states
        # obstacles
        """
        counter = 0
        policy_string = ""
        for row in range(shape[0]):
            for col in range(shape[1]):
                if(p[counter] == -1): policy_string += " *  "
                elif(p[counter] == 3): policy_string += " ^  "
                elif(p[counter] == 0): policy_string += " <  "
                elif(p[counter] == 1): policy_string += " v  "
                elif(p[counter] == 2): policy_string += " >  "
                elif(np.isnan(p[counter])): policy_string += " #  "
                counter += 1
            policy_string += '\n'
        print(policy_string)

    def execute_policy_iteration(self):
        gamma = 0.9
        epsilon = 0.01
        iteration = 0
        T = self.frozen_transition()

        reward = np.array([state_map.get(sublist) for state in frozen_lake.MAPS[self.env.spec._kwargs.get('map_name')] for sublist in state], dtype=float)

        reward[np.where(reward==1)] = 10

        # Generate the first policy randomly
        # NaN=Nothing, -1=Terminal, 0=Up, 1=Left, 2=Down, 3=Right
        p = np.random.randint(2, 3, size=(self.env.observation_space.n)).astype(np.float32)
        p[np.where(reward==-1) or np.where(reward==10)] = -1
        # Utility vectors
        u = np.zeros(self.env.observation_space.n, dtype=float)

        policy_convergence = list()

        while True:
            iteration += 1
            # 1- Policy evaluation
            u_0 = u.copy()
            u = self.return_policy_evaluation(p, u, reward, T, gamma)
            # Stopping criteria
            delta = np.absolute(u - u_0).max()
            policy_convergence.append({'iter': iteration, 'delta': delta})
            if delta < epsilon * (1 - gamma) / gamma: break
            for s in range(self.env.action_space.n):
                if not np.isnan(p[s]) and not p[s] == -1 and not p[s]==10:
                    v = np.zeros((1, self.env.observation_space.n), dtype=float)
                    v[0, s] = 1.0
                    # 2- Policy improvement
                    a = self.return_expected_action(u, T, v)
                    if a != p[s]: p[s] = a
        print("=================== POLICY ITERATION RESULT ==================")
        print("Iterations: " + str(iteration))
        print("Delta: " + str(delta))
        print("Gamma: " + str(gamma))
        print("Epsilon: " + str(epsilon))
        print("===================================================")
        utility_reshape = np.reshape(u, (
        int(np.sqrt(self.env.observation_space.n)), int(np.sqrt(self.env.observation_space.n))))
        print(np.array(utility_reshape, dtype=float))
        print("===================================================")
        self.print_policy(p, shape=(int(np.sqrt(self.env.observation_space.n)), int(np.sqrt(self.env.observation_space.n))))
        print("===================================================")

        print (u)

        return p

    def test_policy(self, policy, value=False):
        grid_policy = np.full((int(np.sqrt(self.env.observation_space.n))  + 2, int(np.sqrt(self.env.observation_space.n))  + 2), -999.0)
        value_row_start = 0
        value_row_end = int(np.sqrt(self.env.observation_space.n))
        for row in range(int(np.sqrt(self.env.observation_space.n)) ):
            row += 1
            grid_policy[row, 1:-1] = policy[value_row_start:value_row_end]
            value_row_start = value_row_end
            value_row_end += int(np.sqrt(self.env.observation_space.n))

        tot_reward = 0
        for i in range(100):
            print('Run: {}'.format(i))
            observation = self.env.reset()
            done = False

            action = policy[observation]
            while not done:
                # self.env.render()
                observation, reward, done, info = self.env.step(int(action))
                # self.env.render()
                if value:
                    action = self.get_action_from_state(0, grid_policy)
                else:
                    action = policy[observation]
                tot_reward += reward
        print('Total Reward: {}'.format(tot_reward))
        return tot_reward

class q_learner():
    def __init__(self, environment):
        self.env = environment

        self.qlearner = ql.QLearner(num_states=self.env.observation_space.n,
                               num_actions=4, dyna=0, verbose=False, rar=0.3, radr=0.99)
        self.qlearner.alpha = 0.1


    def train_model(self):
        for i_episode in range(10000):
            old_table = self.qlearner.q_Table.copy()
            observation = self.env.reset()
            start = True
            done = False
            j = 0
            while not done:
                if start:
                    action = self.qlearner.querysetstate(0)
                    start = False
                observation, reward, done, info = self.env.step(action)
                '''
                Frozen Lake actions:
                0 - Left
                1 - Down
                2 - Right
                3 - Up
                '''
                if not done:
                    reward = -0.01
                if done and observation != self.env.observation_space.n - 1:
                    reward = -1
                elif done and observation == self.env.observation_space.n - 1:
                    reward = 1
                action = self.qlearner.query(observation, reward)
            print (np.abs(self.qlearner.q_Table - old_table).max())

    def test_model(self):
        self.qlearner.num_states = self.env.observation_space.n
        observation = self.env.reset()
        done = False
        q_value = np.argmax(self.qlearner.q_Table[observation])
        tot_reward = 0
        for i in range(100):
            print ('Run: {}'.format(i))
            observation = self.env.reset()
            done = False
            while not done:
                self.env.render()
                observation, reward, done, info = self.env.step(q_value)
                q_value = np.argmax(self.qlearner.q_Table[observation])
            tot_reward += reward
        print ("Q Learner Total Reward: {}".format(tot_reward))

env = gym.make('FrozenLake-v0')

# value_policy = value_iteration_mdp(env)
# value_iter_policy = value_policy.value_iteration()
#
# policy = policy_iteration(env)
# policy_iter_policy = policy.execute_policy_iteration()
#
# value_reward = value_policy.test_policy(value_iter_policy, value= True)
# policy_iter_reward = policy.test_policy(policy_iter_policy)

env_ql = q_learner(env)
env_ql.train_model()
env_ql.test_model()
# print ("Value Iter Reward: {} \n"
#        "Policy Iter Reward: {}".format(value_reward, policy_iter_reward))