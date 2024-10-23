import numpy as np

def compute_optimal_transport(M, r, c, lam, epsilon=1e-8):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the
    Sinkhorn-Knopp algorithm

    Inputs:
        - M : cost matrix (n x m)
        - r : vector of marginals (n, )
        - c : vector of marginals (m, )
        - lam : strength of the entropic regularization
        - epsilon : convergence parameter

    Outputs:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
    """
    n, m = M.shape
    P = np.exp(- lam * M)
    P /= P.sum()
    u = np.zeros(n)
    # normalize this matrix
    while np.max(np.abs(u - P.sum(1))) > epsilon:
        u = P.sum(1)
        P *= (r / u).reshape((-1, 1))#行归r化，注意python中*号含义
        P *= (c / P.sum(0)).reshape((1, -1))#列归c化
    return P, np.sum(P * M)

def main():
    np.random.seed(42)
    a = np.ones([16,])/16
    b = np.ones([4,])/4
    d = np.random.rand(16,4)
    t,v = compute_optimal_transport(d,a,b,1)
    d = list(np.argmin(d, axis=1))
    results_d = {}
    for i in set(d):
        results_d[i] = d.count(i)
    for i,_ in enumerate(t):
        t[i,:] /= t[i,:].sum()
    t = list(np.argmax(t, axis=1))
    results_t = {}
    for i in set(t):
        results_t[i] = t.count(i)

    pass
if __name__ == '__main__':
    main()