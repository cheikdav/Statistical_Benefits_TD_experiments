import numpy as np

from collections import defaultdict

class MC_TD_mult_hor_estimates():
    """
    Class for estimating Monte Carlo (MC) and Temporal Difference (TD) values for multiple horizons and multiple number of samples.
    
    Attributes:
        MC_estimates (list): List of dictionaries to store current MC estimates for each horizon.
        P_estimates (defaultdict): Default dictionary to store current transition probabilities estimates.
        R_estimates (defaultdict): Default dictionary to store current reward estimates.
        N_visits (defaultdict): Default dictionary to store the current number of visits to each state.
        N_first_visits (defaultdict): Default dictionary to store the current number of first visits to each state.
        N_observations (int): Number of current observations.
        archive_MC_estimates (list): List to store archived MC estimates.
        archive_TD_estimates (list): List to store archived TD estimates.
        timestamps (list): List of timestamps, that is the number of samples at which to store the estimates.
        horizons (list): List of horizons, that is the horizons at which the MRP is truncated.
        index (list): List of dictionaries to store an index for each state for each horizon truncation.
        M (object): Object representing the environment.
        Ps (list): List of default dictionaries to store transition probabilities for each horizon.
        Rs (list): List of default dictionaries to store reward estimates for each horizon.
        Ns (list): List of default dictionaries to store the number of visits to each state for each horizon.
    """
    
    def __init__(self, timestamps, horizons, M):
        """
        Initializes the MC_TD_mult_hor_estimates class.
        
        Args:
            timestamps (list): List of timestamps.
            horizons (list): List of horizons.
            M (object): Object representing the environment.
        """
        self.MC_estimates = [defaultdict(float) for _ in horizons]
        self.P_estimates = defaultdict(self.empty_dic)
        self.R_estimates = defaultdict(float)
        self.N_visits = defaultdict(int)
        self.N_first_visits = defaultdict(int)
        self.N_observations = 0
        self.archive_MC_estimates = []
        self.archive_TD_estimates = []
        self.timestamps = timestamps
        self.horizons = horizons
        self.index = [defaultdict(int) for _ in horizons]
        self.M = M
        for i,h in enumerate(horizons):
            for s in M.S:
                if s[1]<=h:
                    self.index[i][s] = len(self.index[i])
        self.Ps = [defaultdict(self.empty_dic) for j in range(len(horizons))]
        self.Rs = [defaultdict(float) for j in range(len(horizons))]
        self.Ns = [defaultdict(int) for _ in horizons]

    def add_trajectory(self, t):
        """
        Adds a trajectory to the estimates.
        
        Args:
            t (list): List representing the trajectory.
        """
        tot_rew = 0
        j = 0
        for i,(s,r,sb) in enumerate(t):
            tot_rew += r

            self.N_visits[s] +=1
            self.P_estimates[s][sb] += 1
            self.R_estimates[s] += r

            self.Ps[j][s][sb] +=1
            self.Rs[j][s] += r
            self.Ns[j][s] += 1
            if sb == -1 or sb[1] == self.horizons[j]:
                self.MC_estimates[j][t[0][0]] += tot_rew
                j += 1
        self.N_first_visits[t[0][0]] += 1
        self.N_observations += 1
        if self.N_observations in self.timestamps:
            TD_estimates = []
            MC_estimates = []
            for j,h in enumerate(self.horizons):
                TD_estimates.append(self.TD_estimation(j,h))
                MC_estimates.append(self.MC_estimation(j))

            self.archive_TD_estimates.append(TD_estimates)
            self.archive_MC_estimates.append(MC_estimates)

    def TD_estimation(self, j,h):
        """
        Performs TD estimation for a given horizon.
        
        Args:
            j (int): Index of the horizon.
            h (int): Horizon value.
        
        Returns:
            defaultdict: Default dictionary containing TD estimates for each state.
        """
        S = len(self.index[j])+1
        P = np.zeros((S, S))
        R = np.zeros(S)
        Ns = defaultdict(int)
        for i in range(j+1):
            added = set()
            for s, succ in self.Ps[i].items():
                if s[1]<h:
                    for s_, num_trans in succ.items():
                        idx = self.index[j][s_] if s_ in self.index[j] else S-1
                        P[self.index[j][s]][idx] += num_trans
                    R[self.index[j][s]] += self.Rs[i][s]
                if s not in added:
                    Ns[s]+= self.Ns[i][s]
                    added.add(s)
        for s in Ns:
            for i in range(S):
                P[self.index[j][s],i]/= Ns[s]
            R[self.index[j][s]]/=Ns[s]
        V = np.linalg.inv(np.eye(S) - P ) @ R
        return defaultdict(float,{s:V[k] for s,k in self.index[j].items() if s in self.M.I})

    def MC_estimation(self, j):
        """
        Performs MC estimation for a given horizon.
        
        Args:
            j (int): Index of the horizon.
        
        Returns:
            defaultdict: Default dictionary containing MC estimates for each state.
        """
        return defaultdict(float,{s: self.MC_estimates[j][s]/self.N_first_visits[s] for s in self.MC_estimates[j]})

    def empty_dic(self):
        """
        Returns an empty default dictionary.
        
        Returns:
            defaultdict: Empty default dictionary.
        """
        return defaultdict(float)