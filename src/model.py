from scipy.stats import norm, bernoulli
import matplotlib.pyplot as plt
import random
import numpy as np

class Congress:
    def __init__(self, beta = 0.5, d_num = 250, r_num = 250, dist = 2, std = 1):
        self.beta = beta
        self.democrats = []
        self.republicans = []
        d_means = norm(loc=dist/2, scale=std).rvs(d_num)
        for i, m in enumerate(d_means):
            self.democrats.append(Person(i, m, 'D'))
        r_means = norm(loc=-1*dist/2, scale=std).rvs(r_num)
        for j, m in enumerate(r_means):
            self.republicans.append(Person(j + d_num, m, 'R'))

    def visualize(self):
        d_means = [rep.ip for rep in self.democrats]
        r_means = [rep.ip for rep in self.republicans]
        plt.hist(d_means, bins=20, color='b', alpha=.5, label='democrats')
        plt.hist(r_means, bins=20, color='r', alpha=.5, label='republicans')
        plt.xlabel('Ideal points')
        plt.ylabel('Count')
        plt.title('Distribution over ideals, per party')
        plt.show()

    def generate_bill(self):
        party = random.choice(['D', 'R'])
        if party == 'D':
            sponsor = random.choice(self.democrats)
        else:
            sponsor = random.choice(self.republicans)
        bill = sponsor.ideals.rvs()
        return sponsor, bill

    def vote_on_bill(self, sponsor, bill):
        d_ayes = 0
        r_ayes = 0
        d_nays = 0
        r_nays = 0
        for r in self.democrats:
            vote = r.vote(sponsor, bill, beta=self.beta)
            if vote == 1:
                d_ayes += 1
            else:
                d_nays += 1
        for r in self.republicans:
            vote = r.vote(sponsor, bill, beta=self.beta)
            if vote == 1:
                r_ayes += 1
            else:
                r_nays += 1
        return d_ayes, d_nays, r_ayes, r_nays

    def run_beta_simulation(self, beta, k=100):
        print('Running simulation for b =', beta)
        self.beta = beta
        bills = []
        scores = []
        for i in range(k):
            sponsor, bill = self.generate_bill()
            d_ayes, d_nays, r_ayes, r_nays = congress.vote_on_bill(sponsor, bill)
            bills.append(bill)
            scores.append(d_ayes+r_ayes)
        return bills, scores

class Person:
    def __init__(self, name, ideal_point, party):
        self.name = name
        self.ip = ideal_point
        self.ideals = norm(loc=ideal_point, scale=1)
        self.max = self.ideals.pdf(self.ip)
        self.party = party
        self.record = []

    def vote(self, sponsor, bill, beta):
        if sponsor.name == self.name:
            return 1
        sim = self.ideals.pdf(bill) / self.max
        aff = 1 if sponsor.party == self.party else -1
        p = (beta * aff) + ((1 - beta) * sim)
        p = np.clip(p, 0, 1.0)
        vote = bernoulli(p).rvs()
        return vote

def report_results(d_ayes, d_nays, r_ayes, r_nays):
    total_ayes = d_ayes + r_ayes
    total_nays = d_nays + r_nays
    if total_ayes > total_nays:
        print('Passed!')
    else:
        print('Did not pass')
    print('Number of ayes: {} (D={}, R={})'.format(total_ayes, d_ayes, r_ayes))
    print('Number of nays: {} (D={}, R={})'.format(total_nays, d_nays, r_nays))

if __name__ == '__main__':
    congress = Congress()
    congress.visualize()
    bills, scores = congress.run_beta_simulation(0, k=300)
    plt.scatter(bills, scores, color='darkviolet', label='b=0', alpha=.8)
    bills, scores = congress.run_beta_simulation(0.5, k=300)
    plt.scatter(bills, scores, color='coral', label='b=.5', alpha=.8)
    bills, scores = congress.run_beta_simulation(0.8, k=300)
    plt.scatter(bills, scores, color='forestgreen', label='b=.8', alpha=.8)

    plt.title('Effect of partisanship on bill success')
    plt.xlabel('Bill ideology')
    plt.ylabel('Number of ayes')
    plt.legend()
    plt.show()
