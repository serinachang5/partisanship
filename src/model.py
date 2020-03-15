import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

class Model:
    def __init__(self, parties, beta, lmb):
        self.beta = beta
        self.lmb = lmb
        self.num_parties = len(parties)
        self.legislators = []
        i = 0
        for c, (n, priors) in enumerate(parties):
            if c == 0:
                self.dim = len(priors)
            else:
                assert(self.dim == len(priors)), 'Must have the same number of priors per party' % self.dim
            X = np.zeros((self.dim, n))  # dim x party_size
            for k in range(self.dim):
                X[k] = norm(loc=priors[k], scale=1).rvs(n)
            for x in X.T:
                self.legislators.append(Legislator(i, x, c))
                i += 1
        print('Done initializing model: N = %d' % (len(self.legislators)))

    def visualize(self):
        if self.dim == 1:
            self._visualize_with_histogram()
        else:
            assert(self.dim == 2), 'Cannot visualize with d > 2'
            self._visualize_with_scatterplot()

    def _visualize_with_histogram(self):
        party2ideologies = {c:[] for c in range(self.num_parties)}
        for l in self.legislators:
            party2ideologies[l.c].append(l.x[0])
        for c, x in party2ideologies.items():
            plt.hist(x, bins=20, alpha=0.5, label='party %d' % c)
        plt.legend()
        plt.xlabel('Ideal points')
        plt.ylabel('Count')
        plt.title('Distribution over ideologies, per party')
        plt.show()

    def _visualize_with_scatterplot(self):
        party2ideologies = {c:[] for c in range(self.num_parties)}
        for l in self.legislators:
            party2ideologies[l.c].append(l.x)
        for c, x in party2ideologies.items():
            d1 = [t[0] for t in x]
            d2 = [t[1] for t in x]
            plt.scatter(d1, d2, alpha=0.5, label='party %d' % c)
        plt.legend()
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('Distribution over ideologiess, per party')
        plt.show()

    def vote_on_bill(self, sponsor, bill):
        likelihoods = np.array([l.get_likelihood(sponsor, bill, beta=self.beta,
                                lmb=self.lmb) for l in self.legislators])
        rands = np.random.uniform(0, 1, len(likelihoods))
        votes = rands < likelihoods
        return np.sum(votes)

    def get_total_similarity(self, bill):
        sims = [l.get_similarity(bill, self.lmb) for l in self.legislators]
        return np.sum(sims)

    def run_beta_simulation(self, beta):
        print('Running simulation for b =', beta)
        self.beta = beta
        bills = []
        votes = []
        for sponsor in self.legislators:  # let each senator sponsor a bill
            bill = sponsor.x
            num_votes = self.vote_on_bill(sponsor, bill)
            bills.append(bill)
            votes.append(num_votes)
        return bills, votes

class Legislator:
    def __init__(self, name, ideal_point, party):
        self.name = name
        self.x = ideal_point
        self.c = party

    def get_likelihood(self, sponsor, bill, beta=0, lmb=.5):
        if sponsor.name == self.name:
            return 1
        aff = self.get_affinity(sponsor)
        sim = self.get_similarity(bill, lmb)
        ll = (beta * aff) + ((1 - beta) * sim)
        return ll

    def get_affinity(self, j):
        if j.c == self.c:
            return 1
        return 0

    def get_similarity(self, bill, lmb):
        d2 = np.sum((self.x - bill) ** 2)
        sim = 1 / ((lmb * d2) + 1)
        return sim

def compare_betas(betas, party_dist, lmb, n=250):
    mu1 = -(party_dist/2)
    mu2 = party_dist/2
    parties = [(n, [mu1]), (n, [mu2])]
    mdl = Model(parties, beta=0, lmb=lmb)
    for b in betas:
        bills, votes = mdl.run_beta_simulation(b)
        plt.scatter(bills, votes, label='b=%s' % b, alpha=.8)

    xmin, xmax = plt.xlim()
    plt.xticks(np.arange(math.floor(xmin), math.ceil(xmax), 1))
    ymin, ymax = plt.ylim()
    plt.vlines([mu1], ymin, ymax, linestyles='dashed', alpha=.5)
    plt.vlines([mu2], ymin, ymax, linestyles='dashdot', alpha=.5)

    plt.title('Effect of beta on bill success (lmb=%s, d=%s)' % (lmb, party_dist))
    plt.xlabel('Bill ideology')
    plt.ylabel('Number of votes in favor')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    compare_betas([0, .5, .8], 4, .25)
