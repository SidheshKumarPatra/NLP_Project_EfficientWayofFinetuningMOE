"""
Bandit block selector for AdaGradSelect-MoE (pure numpy, no torch).

Arms = experts (+ any non-always blocks). Policies:
  bandit   : AdaGradSelect. Epoch 0: eps-greedy (explore = top-k by grad score,
             exploit = Dirichlet(freq + delta) sampling). Epoch >= 1: exploit only.
             eps decays exponentially from eps0 to eps_min over epoch 0.
  gradtopk : always explore (paper's preliminary "top-k by grad norm" experiment)
  random   : uniform random k arms (ablation: is the bandit better than random?)

Freq counts are updated EVERY step (paper Alg. 2 line 17), unlike the sample script.
"""
import math
import numpy as np


class BlockSelector:
    def __init__(self, arms, pct, policy="bandit", eps0=1.0, eps_min=0.01,
                 decay_steps=1000, delta=1.0, seed=0):
        assert policy in ("bandit", "gradtopk", "random")
        self.arms = list(arms)
        self.n = len(self.arms)
        self.k = 0 if pct <= 0 else max(1, min(self.n, int(round(pct / 100.0 * self.n))))  # pct 0 = control: no arms trained
        self.policy = policy
        self.eps0, self.eps_min, self.delta = eps0, eps_min, delta
        self.lam = math.log(eps0 / eps_min) / max(decay_steps, 1)
        self.freq = np.zeros(self.n)
        self.rng = np.random.default_rng(seed)
        self.t = 0            # steps taken in epoch 0 (drives eps decay)
        self.n_explore = 0
        self.n_steps = 0

    def eps(self):
        return max(self.eps_min, self.eps0 * math.exp(-self.lam * self.t))

    def plan(self, epoch):
        """Decide mode BEFORE the forward pass.
        Returns ("explore", None) -> caller must run full backward, then call pick_from_scores
                ("exploit", idx)  -> caller can freeze everything outside idx before backward
        """
        self.n_steps += 1
        if self.policy == "random":
            return "exploit", self._sample(np.full(self.n, 1.0 / self.n))
        if self.policy == "gradtopk":
            self.n_explore += 1
            return "explore", None
        if epoch == 0:
            explore = self.rng.random() < self.eps()
            self.t += 1
        else:
            explore = False
        if explore:
            self.n_explore += 1
            return "explore", None
        return "exploit", self._dirichlet_pick()

    def pick_from_scores(self, scores):
        scores = np.asarray(scores, dtype=np.float64)
        return np.argpartition(-scores, self.k - 1)[: self.k]

    def record(self, idx):
        self.freq[np.asarray(idx, dtype=int)] += 1

    # ---- internals
    def _sample(self, p):
        return self.rng.choice(self.n, size=self.k, replace=False, p=p)

    def _dirichlet_pick(self):
        p = self.rng.dirichlet(self.freq + self.delta)
        p = np.maximum(p, 1e-12)
        p /= p.sum()
        return self._sample(p)

    def stats(self):
        order = np.argsort(-self.freq)
        return {
            "k": self.k, "n_arms": self.n, "n_steps": self.n_steps,
            "n_explore": self.n_explore,
            "freq": {self.arms[i]: int(self.freq[i]) for i in order},
        }