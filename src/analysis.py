import matplotlib.pyplot as plt
from model import Model
import numpy as np
from scipy.stats import ncx2
from scipy.integrate import simps

def get_observed_sim_sf(mdl, bill):
    sims = [l.get_similarity(bill, mdl.lmb) for l in mdl.legislators]
    sims = sorted(sims, reverse=True)
    sf = [i/len(sims) for i, s in enumerate(sims)]
    return sims, sf

def construct_chisq_rv(lmb, mu=None, bill=None, dist=None, dim=None):
    msg = 'Must have mu and bill, or distance between mu and bill and number of dimensions'
    if dist is None:
        assert(mu is not None and bill is not None), msg
        delta = np.array(mu) - np.array(bill)
        nc = np.sum(delta ** 2)
        dof = len(mu)
    else:
        assert(dim is not None), msg
        nc = dist ** 2
        dof = dim
    rv = ncx2(dof, nc)
    return rv, dof, nc

def get_sim_sf(lmb, mu=None, bill=None, dist=None, dim=None, scale=100):
    x2_rv, _, _ = construct_chisq_rv(lmb, mu=mu, bill=bill, dist=dist, dim=dim)
    sims = [s / scale for s in np.arange(1, scale)]
    sf = []
    for s in sims:
        d2 = (1/lmb) * ((1/s) - 1)
        sf.append(x2_rv.cdf(d2))  # cdf of d_ij^2 is sf of s_ij
    return sims, sf

def get_sim_pdf(lmb, mu=None, bill=None, dist=None, dim=None, scale=100):
    x2_rv, _, _ = construct_chisq_rv(lmb, mu=mu, bill=bill, dist=dist, dim=dim)
    sims = [s / scale for s in np.arange(1, scale)]
    pdf = []
    for s in sims:
        d2 = (1/lmb) * ((1/s) - 1)
        pdf.append(x2_rv.pdf(d2))
    return sims, pdf

def plot_sim_curves(lmb, dist, dim):
    sims, sf = get_expected_sim_sf(lmb, dist=dist, dim=dim)
    plt.plot(sims, sf, label='sf')
    sims, pdf = get_expected_sim_pdf(lmb, dist=dist, dim=dim)
    plt.plot(sims, pdf, label='pdf')
    plt.xlabel('q')
    plt.title('Distribution over similarities: lmb=%s, dist=%s, dim=%s' % (lmb, dist, dim))
    plt.grid(alpha=0.5)
    plt.show()

def test_sim_sf(mu, bill, lmb, n=1000):
    party = [(n, mu)]
    mdl = Model(party, beta=0, lmb=lmb)
    sims, sf = get_observed_sim_sf(mdl, bill)
    avg = np.mean(sims)
    print('Observed mean: %.3f' % avg)
    plt.scatter(sims, sf, alpha=.8, s=15, label='observed')

    sims, sf = get_sim_sf(lmb, mu=mu, bill=bill)
    auc = simps(sf, dx=0.01)
    print('Expected mean: %.3f' % auc)
    plt.plot(sims, sf, color='black', label='expected')

    plt.legend()
    plt.xlabel('q')
    plt.ylabel('p(sim_ij > q)')
    plt.title('Cum. distribution over similarities: mu=%s, bill=%s, lmb=%s' % (mu, bill, lmb))
    plt.grid(alpha=0.5)
    plt.show()

def get_party_tradeoff(lmb, party_dist, dim, halfway=False, scale=100):
    dist_range = np.arange(1, .5 * scale) if halfway else np.arange(1, scale)
    D = [party_dist * d/scale for d in dist_range]
    E_a = []
    E_b = []
    total = []
    for d_a in D:
        _, sf_a = get_sim_sf(lmb, dist=d_a, dim=dim)
        sim_a = simps(sf_a, dx=0.01)
        E_a.append(sim_a)
        d_b = party_dist - d_a
        _, sf_b = get_sim_sf(lmb, dist=d_b, dim=dim)
        sim_b = simps(sf_b, dx=0.01)
        E_b.append(sim_b)
        total.append((sim_a + sim_b)/2)
    return D, E_a, E_b, total

def analyze_party_tradeoff(D, E_a, total, mid_idx):
    first_deriv = np.gradient(E_a)
    second_deriv = np.gradient(first_deriv)
    inf_pt = None
    for i, dd in enumerate(second_deriv):
        if dd > 0:
            inf_pt = (D[i]+D[i-1])/2  # sign changed between these points
            break
    midpoint = D[mid_idx]
    midpoint_util = total[mid_idx]
    argmax = np.argmax(total)
    opt = D[argmax]
    opt_util = total[argmax]
    return inf_pt, midpoint, midpoint_util, opt, opt_util

def plot_party_tradeoff(lmb, party_dist, dim, gran=2, print_analysis=False):
    scale = 10 ** gran
    D, E_a, E_b, total = get_party_tradeoff(lmb, party_dist, dim, scale=scale)
    if print_analysis:
        mid_idx = int((scale / 2) - 1)
        inf_pt, midpoint, midpoint_util, opt, opt_util = analyze_party_tradeoff(D, E_a, total, mid_idx)
        print('Inflection point: %.5f' % inf_pt)
        print('Midpoint: %.5f' % midpoint)
        print('Avg util at midpoint: %.5f' % midpoint_util)
        print('Optimal point: %.5f' % opt)
        print('Optimal avg util: %.5f' % opt_util)
        # _, _, _, ind_total = get_party_individual_tradeoff(lmb, party_dist, dim, scale=scale)
        # argmax = np.argmax(ind_total)
        # print('Optimal point for individual: %.5f' % D[argmax])
    plt.plot(D, E_a, label='party A')
    plt.plot(D, E_b, label='party B')
    plt.plot(D, total, label='total')
    plt.legend()
    plt.title('lmb=%s, dim=%s, party_dist=%s' % (lmb, dim, party_dist))
    plt.xlabel('Distance of bill from party A')
    plt.ylabel('Avg utility')
    plt.show()

def get_party_individual_tradeoff(lmb, party_dist, dim, halfway=False, scale=100):
    marker = 2 / lmb  # q = 0.5
    dist_range = np.arange(1, .5 * scale) if halfway else np.arange(1, scale)
    D = [party_dist * d/scale for d in dist_range]
    F_a = []
    F_b = []
    total = []
    for d_a in D:
        rv_a, _, _ = construct_chisq_rv(lmb, dist=d_a, dim=dim)
        f_a = rv_a.cdf(marker)
        F_a.append(f_a)
        d_b = party_dist - d_a
        rv_b, _, _ = construct_chisq_rv(lmb, dist=d_b, dim=dim)
        f_b = rv_b.cdf(marker)
        F_b.append(f_b)
        total.append((f_a + f_b)/2)
    return D, F_a, F_b, total

def plot_d_curves(lmb, dim, dists):
    for dist in dists:
        sims, sf = get_sim_sf(lmb, dist=dist, dim=dim, scale=1000)
        plt.plot(sims, sf, label='d=%.1f' % dist)
    plt.legend()
    plt.xlabel('q')
    plt.ylabel('p(sim_ij > q)')
    plt.title('lmb=%s, dim=%s' % (lmb, dim))
    plt.show()

def plot_lambda_versus_max_util(lmbs, party_dists, dim):
    for dist in party_dists:
        max_utils = []
        for lmb in lmbs:
            D, E_a, E_b, total = get_party_tradeoff(lmb, dist, dim, halfway=True)
            max_utils.append(max(total))
        plt.plot(lmbs, max_utils, label='d=%d' % dist)
    plt.xlabel('Lambda')
    plt.ylabel('Optimal utility')
    plt.legend()
    plt.title('dim=%s' % dim)
    plt.grid(alpha=.3)
    plt.show()

def plot_dist_versus_max_util(lmbs, party_dists, dim):
    for lmb in lmbs:
        max_utils = []
        for dist in party_dists:
            D, E_a, E_b, total = get_party_tradeoff(lmb, dist, dim, halfway=True)
            max_utils.append(max(total))
        plt.plot(party_dists, max_utils, label='lmb=%.2f' % lmb)
    plt.xlabel('Party distance')
    plt.ylabel('Optimal utility')
    plt.legend()
    plt.title('dim=%s' % dim)
    plt.grid(alpha=.3)
    plt.show()

def plot_dist_versus_optimal_bill_rel_pos(party_dists):
    # TO DO: fix, lambda need to be varied
    lmb = 0.5
    for dim in range(1, 5):
        pos = []
        nonmid = False
        for dist in party_dists:
            D, E_a, E_b, total = get_party_tradeoff(lmb, dist, dim, halfway=True)
            argmax = np.argmax(total)
            rel_pos = D[argmax] / dist
            pos.append(rel_pos)
            if nonmid is False and rel_pos < .5:
                print('dim = %s: first dist with non-midpoint = %.2f' % (dim, dist))
                nonmid = True
        plt.plot(party_dists, pos, label='dim=%d' % dim)
    plt.xlabel('Party distance')
    plt.ylabel('Rel. position of optimal bill')
    plt.legend()
    plt.title('dim=%s' % dim)
    plt.grid(alpha=.3)
    plt.show()

def compare_individual_vs_avg(lmb, dim, dists, scale=1000):
    for dist in dists:
        sims, sf = get_sim_sf(lmb, dist=dist, dim=dim, scale=scale)
        exp_sim = simps(sf, dx=1/scale)
        mid_idx = int((scale / 2) - 1)
        sf_at_mid = sf[mid_idx]
        print('Dist = %.3f -> avg sim: %.3f, p(sim_ij > 0.5) = %.3f' % (dist, exp_sim, sf_at_mid))

if __name__ == '__main__':
    test_sim_sf([0, 0], [1e-10, 1e-10], 1)
    # plot_expected_sim_curves(0.5, 2, 1)
    # plot_party_tradeoff(0.5, 3, 1, print_analysis=True)
    # plot_party_individual_tradeoff(0.5, 2, 1)
    # dists = [i / 4 for i in range(1, 9)]  # [1, 2, 3, 4, 5]  #
    # plot_d_curves(.5, 1, dists)
    # lmbs = [.5, 1, 1.5, 2]  # [i / 10 for i in range(1, 21)]
    # plot_lambda_versus_optimal_bill(lmbs, dists, 1)
    # plot_dist_versus_optimal_bill(lmbs, dists, 1)
    # plot_dist_versus_optimal_bill_rel_pos(dists)
    # compare_individual_vs_avg(.25, 1, dists)
