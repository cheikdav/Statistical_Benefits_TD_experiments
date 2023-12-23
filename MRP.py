import numpy as np
import matplotlib.pyplot as plt
import copy

from collections import defaultdict

class Reward:
    def __init__(self, generator = None, mean = 0, var = 0, add = 0):
        """
        Initialize a Reward object.

        Parameters:
        - generator (function): A function that generates a reward with mean mean and variance var.
        - mean (float): The mean value of the reward.
        - var (float): The variance of the reward.
        - add (float): An additional constant value to add to the reward.

        Returns:
        None
        """
        self.generator = generator
        self.add = add
        self.m = mean + add
        self.var = var

    def sample(self):
        """
        Sample a reward.

        Returns:
        float: The sampled reward.
        """
        if self.generator is None:
            return self.add
        return self.generator() + self.add

    def mean(self):
        """
        Get the mean value of the reward.

        Returns:
        float: The mean value of the reward.
        """
        return self.m

    def variance(self):
        """
        Get the variance of the reward.

        Returns:
        float: The variance of the reward.
        """
        return self.var

    def __eq__(self, other):
        """
        Check if two Reward objects are equal.

        Parameters:
        - other (Reward): The other Reward object to compare.

        Returns:
        bool: True if the two Reward objects have same mean and variance, False otherwise.
        """
        return self.m == other.m and self.var == other.var


class MRP:
    def __init__(self, S, P, R, I, g, ν, rnd_gen = None):
        """
        Initialize a Markov Reward Process (MRP) object.

        Parameters:
        - S (list): The state space, excluding the end state.
        - P (dict): The transition probability. Dict s->([next states], [transition probability]).
                    Probabilities should sum to less than 1 (<1). If a state has no successor,
                    s can either be absent from P or be associated with an empty tuple.
        - R (dict): The rewards. R[s,s'] should be a Reward object.
        - I (list): The set of initial states. Should be a subset of S.
        - g (int): The end state, not part of S.
        - ν (list): The initial distribution over I (as a vector of probabilities).
        - rnd_gen (numpy.random.Generator): The random number generator.

        Returns:
        None
        """
        if rnd_gen == None:
            rnd_gen = np.random.default_rng()
        self.rnd_gen = rnd_gen
        self.S = S
        self.P = {s: (list(P[s][0]), list(P[s][1])) if len(P[s]) > 0 else () for s in P}
        self.R = R
        self.I = I
        self.g = g
        self.ν = list(ν)
        self.s = self.I[self.rnd_gen.choice(len(self.I), p=self.ν)]
        self.number = {}
        self.next_trans = {s:[] for s in self.S}
        self.id_next_trans = {s:0 for s in self.S}
        for i, s in enumerate(self.I):
            self.number[s] = i
        self.compute_matrices()

    def __eq__(self, other):
        """
        Check if two MRP objects are equal.

        Parameters:
        - other (MRP): The other MRP object to compare.

        Returns:
        bool: True if the two MRP objects are equal, False otherwise.
        """
        if set(self.S) != set(other.S):
            return False
        if self.P != other.P:
            return False
        if self.R != other.R:
            return False
        if self.I != other.I:
            return False
        if self.ν != other.ν:
            return False
        return True

    def reseed_gen(self, seed):
        """
        Reset the seed of the random number generator.

        Parameters:
        - seed (int): The seed value.

        Returns:
        None
        """
        self.rnd_gen.__setstate__(np.random.default_rng(seed).__getstate__())

    def reinit(self):
        """
        Reinitialize the MRP by setting the current state to a new initial state.

        Returns:
        None
        """
        self.s = self.I[self.rnd_gen.choice(len(self.I), p=self.ν)]

    def get_state(self):
        """
        Get the current state of the MRP.

        Returns:
        int: The current state.
        """
        return self.s

    def compute_matrices(self):
        """
        Compute the matrix representation of P and rho from the dictionary representation. Useful for computing the TD estimates.

        Returns:
        None
        """
        self.index_map = {}
        for i, s in enumerate(self.S + [self.g]):
            self.index_map[s] = i
        self.matrix_P = np.zeros((len(self.S) + 1, len(self.S) + 1))
        self.rho = np.zeros(len(self.S) + 1)
        p_exits = {}
        for s in self.S:
            p_exit = 1
            if s in self.P:
                for (s_, p) in zip(*self.P[s]):
                    p_exit -= p
                    self.matrix_P[self.index_map[s], self.index_map[s_]] = p
                    self.rho[self.index_map[s]] += p * self.R[(s, s_)].mean()
            self.rho[self.index_map[s]] += p_exit * self.R[(s, self.g)].mean()
            p_exits[s] = p_exit
        self.matrix_eta = np.linalg.inv(np.eye(len(self.S) + 1) - self.matrix_P)
        self.matrix_V = self.matrix_eta @ self.rho
        self.V = {s: self.matrix_V[self.index_map[s]] for s in self.S + [self.g]}
        self.eta = defaultdict(float, {(s, s_): self.matrix_eta[self.index_map[s], self.index_map[s_]] for s in self.I for s_ in self.S})
        self.one_step_vars = defaultdict(float, {s: p_exits[s] * (((self.V[s] - self.R[(s, self.g)].mean()) ** 2 + self.R[(s, self.g)].variance())) + sum([p * ((self.V[s] - self.R[(s, s_)].mean() - self.V[s_]) ** 2 + self.R[(s, s_)].variance())
                                      for (s_, p) in zip(*self.P[s])]) for s in self.S if s in self.P})

    def transition(self):
        """
        Perform a transition from the current state to the next state.

        Returns:
        tuple: The next state and the reward.
        """
        next_s = self.g
        l = len(self.next_trans[self.s])
        if l == 0 or self.id_next_trans[self.s] == l:
            self.id_next_trans[self.s] = 0
            if self.s in self.P and self.P[self.s]:
                ids = self.rnd_gen.choice(range(len(self.P[self.s][0])), p=self.P[self.s][1], size=100)
                self.next_trans[self.s] = [self.P[self.s][0][i] for i in ids]
            else:
                self.next_trans[self.s] = [self.g for _ in range(100)]
        next_s = self.next_trans[self.s][self.id_next_trans[self.s]]
        self.id_next_trans[self.s] += 1
        reward = self.R[(self.s, next_s)].sample()
        self.s = next_s
        return (next_s, reward)

    def generate_path(self):
        """
        Generate a path, starting from a state from I sampled from ν.

        Returns:
        list: The generated path.
        """
        d = []
        self.reinit()
        while True:
            s = self.get_state()
            (next_s, r) = self.transition()
            d.append((s, r, next_s))
            if next_s == self.g:
                return d

    def asymptotic_MC_variance(self, s):
        """
        Compute the asymptotic variance of the MC estimate for a given state.

        Parameters:
        - s (int): The state.

        Returns:
        float: The asymptotic Monte Carlo variance.
        """
        return sum([self.eta[(s, s_)] * self.one_step_vars[s_] for s_ in self.index_map]) / self.ν[self.number[s]]

    def asymptotic_TD_variance(self, s):
        """
        Compute the asymptotic variance of the TD estimate for a given state.

        Parameters:
        - s (int): The state.

        Returns:
        float: The asymptotic TD variance.
        """
        return sum([self.eta[(s, s_)] ** 2 * self.one_step_vars[s_] / sum([self.ν[self.number[st]] * self.eta[(st, s_)] for st in self.I]) for s_ in self.index_map if self.eta[(s, s_)] != 0])

    def asymptotic_diff_TD_variance(self, s, sp):
        """
        Compute the asymptotic variance of the difference of TD estimates for two given states.

        Parameters:
        - s (int): The first state.
        - sp (int): The second state.

        Returns:
        float: The asymptotic differential TD variance.
        """
        return sum([(self.eta[(s, s_)] - self.eta[(sp, s_)]) ** 2 * self.one_step_vars[s_] / sum([self.ν[self.number[st]] * self.eta[(st, s_)] for st in self.I]) for s_ in self.index_map if self.eta[(s, s_)] - self.eta[(sp, s_)] != 0])

    def asymptotic_agg_TD_variance(self, s_ini, index):
        """
        Compute the asymptotic TD variance for a given initial state when using state aggregation.

        Parameters:
        - s_ini (int): The initial state.
        - index (dict): The index mapping. States mapped to the same index are aggregated.

        Returns:
        float: The asymptotic aggregated TD variance.
        """
        S = max(index.values()) + 1
        groups = [[] for _ in range(S)]
        for s, k in index.items():
            groups[k].append(s)

        P = np.zeros((S, S))
        tot_etas = np.zeros(S)
        for num, g in enumerate(groups):
            tot_eta = 0
            for s in g:
                eta = sum([self.ν[self.number[st]] * self.eta[(st, s)] for st in self.I])
                tot_eta += eta
                if s in self.P:
                    for (s_, p) in zip(*self.P[s]):
                        P[num, index[s_]] += eta * p

            tot_etas[num] = tot_eta
            for i in range(S):
                P[num, i] /= tot_eta

        matrix_eta = np.linalg.inv(np.eye(S) - P)
        return sum([matrix_eta[(index[s_ini], index[s_])] ** 2 * sum([self.ν[self.number[st]] * self.eta[(st, s_)] for st in self.I]) * self.one_step_vars[s_] / tot_etas[index[s_]] ** 2 for s_ in self.S])


class RandomLayer(MRP):
    def __init__(self, width, depth, p=0, rnd_gen=None):
        """
        Initialize a MRP with a random layer structure. States transitions from one layer to another.
        When the probability of backward arc is not 0, some backwards arc may be present.

        Parameters:
        - width (int): The width of the layer.
        - depth (int): The depth of the layer.
        - p (float): The probability of adding a backward arc.
        - rnd_gen (numpy.random.Generator): The random number generator.

        Returns:
        None
        """
        if rnd_gen is None:
            rnd_gen = np.random.default_rng()
        self.depth = depth
        self.width = width
        ν = rnd_gen.dirichlet(np.ones(width))
        states = [(w, d) for w in range(width) for d in range(depth)]
        self.R_list = []
        self.P_list = []
        self.successor = {}
        for d in range(self.depth):
            for w in range(self.width):
                self.successor[(w, d)] = [(i, d + 1) for i in range(width)] if d + 1 < depth else [-1]
                if rnd_gen.random() < p:
                    (wb, db) = (rnd_gen.integers(0, width), rnd_gen.integers(0, d + 1))
                    self.successor[(w, d)].append((wb, db))
            self.R_list.append(defaultdict(Reward,{((w,d),s):Reward(self.unif_reward, 0, 1/3, rnd_gen.uniform(-1,1)) for w in range(width) for s in self.successor[(w,d)]}))
            self.P_list.append({(w, d): ([s for s in self.successor[(w, d)]], rnd_gen.dirichlet(np.ones(len(self.successor[(w, d)])))) if d + 1 < depth else () for w in range(width)})

        R = defaultdict(Reward)
        P = {}
        for r in self.R_list:
            R.update(r)
        for p in self.P_list:
            P.update(p)

        MRP.__init__(self, states, P, R, [(w, 0) for w in range(width)], -1, ν, rnd_gen=rnd_gen)

    def remove_cycles(self):
        """
        Remove cycles from the RandomLayer. Useful to analyse how cycles impact the variance of estimators.

        Returns:
        RandomLayer: The RandomLayer object without cycles.
        """
        P = self.P
        R = self.R
        for d in range(self.depth):
            for w in range(self.width):
                if self.successor[(w, d)][-1] != -1 and P[(w, d)] != () and self.successor[(w, d)][-1][1] <= d:
                    del R[((w, d), self.successor[(w, d)][-1])]
                    trans = P[(w, d)]
                    trans = (trans[0][:-1], trans[1][:-1] / sum(trans[1][:-1]))
                    P[(w, d)] = trans
        M = copy.deepcopy(self)
        M.P = P
        M.R = R
        return M

    def unif_reward(self):
        """
        Generate a uniformly distributed reward.

        Returns:
        float: The uniformly distributed reward.
        """
        return self.rnd_gen.uniform(-1, 1)

    def compute_matrices(self, horizon=None):
        """
        Compute the matrix representation of P and rho from the dictionary representation for a fixed horizon:
        Truncate the MRP to the given horizon and compute the matrices in this truncated MRP.

        Parameters:
        - horizon (int): The horizon value.

        Returns:
        None
        """
        self.index_map = {}
        i = 0
        for s in self.S:
            if horizon is None or s[1] < horizon:
                self.index_map[s] = i
                i += 1
        size = len(self.index_map)
        self.matrix_P = np.zeros((size + 1, size + 1))
        self.rho = np.zeros(size + 1)
        p_exits = {}
        for s in self.S:
            if horizon is not None and s[1] >= horizon:
                continue
            p_exit = 1
            if s in self.P:
                for (s_, p) in zip(*self.P[s]):
                    p_exit -= p
                    idx = self.index_map[s_] if s_ in self.index_map else size
                    self.matrix_P[self.index_map[s], idx] += p
                    self.rho[self.index_map[s]] += p * self.R[(s, s_)].mean()
            self.rho[self.index_map[s]] += p_exit * self.R[(s, self.g)].mean()
            p_exits[s] = p_exit
        self.matrix_eta = np.linalg.inv(np.eye(size + 1) - self.matrix_P)
        self.matrix_V = self.matrix_eta @ self.rho
        self.V = defaultdict(float, {s: self.matrix_V[self.index_map[s]] for s in self.index_map})
        self.eta = defaultdict(float, {(s, s_): self.matrix_eta[self.index_map[s], self.index_map[s_]] for s in self.I for s_ in self.index_map})
        self.one_step_vars = defaultdict(float, {s: p_exits[s] * (((self.V[s] - self.R[(s, self.g)].mean()) ** 2 + self.R[(s, self.g)].variance())) + sum([p * ((self.V[s] - self.R[(s, s_)].mean() - self.V[s_]) ** 2 + self.R[(s, s_)].variance())
                                      for (s_, p) in zip(*self.P[s])]) for s in self.index_map if s in self.P})
