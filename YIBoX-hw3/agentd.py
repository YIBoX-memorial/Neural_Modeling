import numpy as np
from scipy import stats # for gaussian noise
from environment import Environment

class DynaAgent(Environment):

    def __init__(self, alpha, gamma, epsilon):

        '''
        Initialise the agent class instance
        Input arguments:
            alpha   -- learning rate \in (0, 1]
            gamma   -- discount factor \in (0, 1)
            epsilon -- controls the influence of the exploration bonus
        '''

        self.alpha   = alpha
        self.gamma   = gamma 
        self.epsilon = epsilon

        return None

    def init_env(self, **env_config):

        '''
        Initialise the environment
        Input arguments:
            **env_config -- dictionary with environment parameters
        '''

        Environment.__init__(self, **env_config)
        # print(self.__dir__())
        return None

    def _init_q_values(self):

        '''
        Initialise the Q-value table
        '''

        self.Q = np.zeros((self.num_states, self.num_actions))

        return None

    def _init_experience_buffer(self):

        '''
        Initialise the experience buffer
        '''

        self.experience_buffer = np.zeros((self.num_states*self.num_actions, 4), dtype=int)
        for s in range(self.num_states):
            for a in range(self.num_actions):
                self.experience_buffer[s*self.num_actions+a] = [s, a, 0, s]

        return None

    def _init_history(self):

        '''
        Initialise the history
        '''

        self.history = np.empty((0, 4), dtype=int)

        return None
    
    def _init_action_count(self):

        '''
        Initialise the action count
        '''

        self.action_count = np.zeros((self.num_states, self.num_actions), dtype=int)

        return None

    def _update_experience_buffer(self, s, a, r, s1):

        '''
        Update the experience buffer (world model)
        Input arguments:
            s  -- initial state
            a  -- chosen action
            r  -- received reward
            s1 -- next state
        '''

        # complete the code
        self.experience_buffer[s*self.num_actions+a,:] = np.asarray((s,a,r,s1))
        return None

    def _update_qvals(self, s, a, r, s1, bonus=False):

        '''
        Update the Q-value table
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
            bonus -- True / False whether to use exploration bonus or not
        '''

        # complete the code
        self.Q[s,a] = self.Q[s,a] + self.alpha*(r + self.epsilon * bonus * np.sqrt(self.action_count[s, a]) + self.gamma*np.max(self.Q[s1,:]) - self.Q[s,a])
        return None

    def _update_action_count(self, s, a):

        '''
        Update the action count
        Input arguments:
            Input arguments:
            s  -- initial state
            a  -- chosen action
        '''

        # complete the code
        self.action_count += 1
        self.action_count[s,a] = 0
        return None

    def _update_history(self, s, a, r, s1):

        '''
        Update the history
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
        '''

        self.history = np.vstack((self.history, np.array([s, a, r, s1])))

        return None

    def _policy(self, s):

        '''
        Agent's policy 
        Input arguments:
            s -- state
        Output:
            a -- index of action to be chosen
        '''

        # complete the code
        newQ = self.Q[s,:] + self.epsilon * np.sqrt(self.action_count[s,:])
        # print(newQ)
        if self.epsilon >= 1:
            if np.random.uniform() < 0.05:
                return np.random.randint(0,self.num_actions)
            else:
                return np.argmax(newQ)
        else:
            # print(np.where(newQ == newQ.max()))
            return np.random.choice(np.where(newQ == newQ.max())[0])
        return None

    def _plan(self, num_planning_updates):

        '''
        Planning computations
        Input arguments:
            num_planning_updates -- number of planning updates to execute
        '''

        # complete the code
        for _ in range(num_planning_updates):
            ind = np.random.randint(self.experience_buffer.shape[0])
            s,a,r,s1 = self.experience_buffer[ind]
            self._update_qvals(s, a, r, s1, bonus=True)
        return None

    def get_performace(self):

        '''
        Returns cumulative reward collected prior to each move
        '''

        return np.cumsum(self.history[:, 2])

    def pp(self, num_trials, reset_agent=True, num_planning_updates=None):
        print("pp")
        return None
    def simulate(self, num_trials, reset_agent=True, num_planning_updates=None):

        '''
        Main simulation function
        Input arguments:
            num_trials           -- number of trials (i.e., moves) to simulate
            reset_agent          -- whether to reset all knowledge and begin at the start state
            num_planning_updates -- number of planning updates to execute after every move
        '''
        # print("Simulating {} trials...".format(num_trials))
        if reset_agent:
            self._init_q_values()
            self._init_experience_buffer()
            self._init_action_count()
            self._init_history()

            self.s = self.start_state

        # print('Simulating {} trials...'.format(num_trials))
        for _ in range(num_trials):

            # choose action
            a  = self._policy(self.s)
            # get new state
            s1 = np.random.choice(np.arange(self.num_states), p=self.T[self.s, a, :])
            # receive reward
            r  = self.R[self.s, a]
            # learning
            self._update_qvals(self.s, a, r, s1, bonus=False)
            # update world model 
            self._update_experience_buffer(self.s, a, r, s1)
            # reset action count
            self._update_action_count(self.s, a)
            # update history
            self._update_history(self.s, a, r, s1)
            # print(self.s, a, r, s1)
            # plan
            if num_planning_updates is not None:
                self._plan(num_planning_updates)

            if s1 == self.goal_state:
                self.s = self.start_state
            else:
                self.s = s1

        return None
    
class TwoStepAgent:

    def __init__(self, alpha1, alpha2, beta1, beta2, lam, w, p):

        '''
        Initialise the agent class instance
        Input arguments:
            alpha1 -- learning rate for the first stage \in (0, 1]
            alpha2 -- learning rate for the second stage \in (0, 1]
            beta1  -- inverse temperature for the first stage
            beta2  -- inverse temperature for the second stage
            lam    -- eligibility trace parameter
            w      -- mixing weight for MF vs MB \in [0, 1] 
            p      -- perseveration strength
        '''

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1  = beta1
        self.beta2  = beta2
        self.lam    = lam
        self.w      = w
        self.p      = p

        self.num_states  = 3
        self.num_actions = 2

        return None
    def _init_q_values(self):
        self.QMF = np.zeros((self.num_states * self.num_actions))
        self.QMB = np.zeros((2))
        self.Qnet = np.zeros((self.num_states * self.num_actions))
        return None
    def _init_history(self):

        '''
        Initialise history to later compute stay probabilities
        '''

        self.history = np.empty((0, 3), dtype=int)

        return None
    
    def _update_history(self, a, s1, r1):

        '''
        Update history
        Input arguments:
            a  -- first stage action
            s1 -- second stage state
            r1 -- second stage reward
        '''

        self.history = np.vstack((self.history, [a, s1, r1]))

        return None
    def _init_reward(self):

        '''
        Initialise rewards uniformly and determine boundaries
        '''

        self.rewards = np.random.uniform(0.25,0.75,4)
        self.boundaries = [0.25,0.75]

        return None
    def get_reward(self,s,a):
        p = self.rewards[2*(s-1)+a]
        r = np.random.choice((0,1), p=(1-p, p))
        return r
    def update_rewards(self):
        self.rewards += np.random.normal(loc=0, scale=0.025, size=4)
        #check if rewards are outside the boundaries and make the boundary reflective
        for i, reward in enumerate(self.rewards):
            if (reward < self.boundaries[0]):
                difference = self.boundaries[0] - reward
                self.rewards[i] += 2*difference
            elif (reward>self.boundaries[1]):
                difference = self.boundaries[1] - reward
                self.rewards[i] += 2*difference
    def _init_transition_matrix(self):
        self.trans_mtx = [[0.7,0.3], [0.3,0.7]]
        self.current_trans_mtx = self.trans_mtx[0]
        return None
    def _init_track_state_transition(self):
        self.trans_1 = 0
        self.trans_2 = 0
        self.last_a = -1
    def _track_state_transitions(self, a, s1):
        if (s1==1 and a==0) or (s1==2 and a==1):
            self.trans_1 += 1
        elif (s1==2 and a==0) or (s1==1 and a==1):
            self.trans_2 += 1
    def _update_q_mb(self, s, a):
        if self.trans_1 > self.trans_2:
            self.current_trans_mtx = self.trans_mtx[0]
        else:
            self.current_trans_mtx = self.trans_mtx[1]
        s_a = s * self.num_actions + a
        prob1 = self.current_trans_mtx[a]
        prob2 = 1 - prob1

        best_q1 = np.max(self.QMF[2:4]) 
        best_q2 = np.max(self.QMF[4:])
        self.QMB[s_a] = prob1*best_q1 + prob2*best_q2
    def _update_q_mf(self, s, a, r, s1, a1, r1):
        if s == 0:
            s_a_old = s * self.num_actions + a
            s_a_new = s1 * self.num_actions + a1
            alpha = self.alpha1
            delta = r1 - self.QMF[s_a_new]
            self.QMF[s_a_old] += alpha * delta * self.lam
            delta = r + self.QMF[s_a_new] - self.QMF[s_a_old]
            self.QMF[s_a_old] += alpha * delta
        else:
            s_a = s * self.num_actions + a
            alpha = self.alpha2
            delta = r - self.QMF[s_a]
            self.QMF[s_a] += alpha * delta
    def _update_q_net(self, s, a):
        s_a = s * self.num_actions + a
        self.Qnet[s_a] = self.w * self.QMF[s_a] + (1-self.w) * self.QMB[s_a]
    def _policy(self, s):
        '''
        Agent's policy 
        Input arguments:
            s -- state
        Output:
            a -- index of action to be chosen
        '''
        exp_terms = np.zeros(2)
        # rep is 1 if we are in the first stage and the current action is the same
        # as in the last trial. rep is 0 else
        for a in range(self.num_actions):
            if s == 0:
                beta = self.beta1
                rep = self.last_a == a
            else:
                beta = self.beta2
                rep = 0
            s_a = s * self.num_actions + a
            exp_terms[a] = np.exp(beta * (self.Qnet[s_a] + self.p * rep))
        policy =  exp_terms / exp_terms.sum()
        a = np.random.choice(np.arange(2), p=policy)
        return a
    
    def get_next_state(self,s,a):
        if s == 0:
            prob1 = self.trans_mtx[0][a]
            p = [prob1, 1-prob1]
            p = np.round(p,1)
            new_s = np.random.choice([1,2], p=p)
        else:
            new_s = 3
        return new_s

    def get_stay_probabilities(self):

        '''
        Calculate stay probabilities
        '''

        common_r      = 0
        num_common_r  = 0
        common_nr     = 0
        num_common_nr = 0
        rare_r        = 0
        num_rare_r    = 0
        rare_nr       = 0
        num_rare_nr   = 0

        num_trials = self.history.shape[0]
        for idx_trial in range(num_trials-1):
            a, s1, r1 = self.history[idx_trial, :]
            a_next    = self.history[idx_trial+1, 0]

            # common
            if (a == 0 and s1 == 1) or (a == 1 and s1 == 2):
                # rewarded
                if r1 == 1:
                    if a == a_next:
                        common_r += 1
                    num_common_r += 1
                else:
                    if a == a_next:
                        common_nr += 1
                    num_common_nr += 1
            else:
                if r1 == 1:
                    if a == a_next:
                        rare_r += 1
                    num_rare_r += 1
                else:
                    if a == a_next:
                        rare_nr += 1
                    num_rare_nr += 1

        return np.array([common_r/num_common_r, rare_r/num_rare_r, common_nr/num_common_nr, rare_nr/num_rare_nr])

    def simulate(self, num_trials):

        '''
        Main simulation function
        Input arguments:
            num_trials -- number of trials to simulate
        '''
            
        # complete the code
        self._init_q_values()
        self._init_track_state_transition()
        self._init_history()
        self._init_transition_matrix()
        self._init_reward()

        for _ in range(num_trials):
            s1 = 0
            a1 = self._policy(s1)
            r1 = 0
            # get new state
            s2 = self.get_next_state(s1,a1)

            self._track_state_transitions(a1,s2)
            self.last_a = a1

            a2 = self._policy(s2)

            r2 = self.get_reward(s2,a2)

            self._update_q_mf(s1, a1, r1, s2, a2, r2)
            self._update_q_mf(s2, a2, r2, _, _, _)
            self._update_q_mb(s1,a1)
            self._update_q_mb(s1,1-a1)
            self._update_q_net(s1,a1)
            self._update_q_net(s1,1-a1)
            self.Qnet[2:] = self.QMF[2:]

            self._update_history(a1, s2, r2)
            self.update_rewards()
        return None