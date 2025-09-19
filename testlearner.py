import sys, time, numpy as np, matplotlib.pyplot as plt
import DTLearner as dt, RTLearner as rt, BagLearner as bl

def load_xy(csv_path):
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    X, y = data[:, 1:-1], data[:, -1]
    return X, y

def split_train_test(X, y, train_frac=0.6, seed=None):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    ntr = int(np.floor(train_frac * X.shape[0]))
    return X[idx[:ntr]], y[idx[:ntr]], X[idx[ntr:]], y[idx[ntr:]]

def rmse(yh, y): return float(np.sqrt(np.mean((yh - y) ** 2)))
def mae(yh, y):  return float(np.mean(np.abs(yh - y)))
def r2(yh, y): 
    ss_res = np.sum((y - yh)**2); ss_tot = np.sum((y - np.mean(y))**2)
    return float(1.0 - ss_res/ss_tot) if ss_tot>0 else 0.0

def exp1_dt_overfitting(X, y, seed=None):
    Xtr, ytr, Xte, yte = split_train_test(X, y, seed=seed)
    leaf_sizes = [1,2,4,8,16,32,64,128]
    tr, te = [], []
    for ls in leaf_sizes:
        m = dt.DTLearner(leaf_size=ls); m.add_evidence(Xtr, ytr)
        tr.append(rmse(m.query(Xtr), ytr)); te.append(rmse(m.query(Xte), yte))
    plt.figure(); plt.plot(leaf_sizes, tr, marker='o', label='Train RMSE')
    plt.plot(leaf_sizes, te, marker='o', label='Test RMSE')
    plt.xscale('log', base=2); plt.xlabel('leaf_size (log2)'); plt.ylabel('RMSE')
    plt.title('Exp-1: DTLearner Overfitting vs leaf_size'); plt.legend(); plt.grid(True, linewidth=0.3)
    plt.savefig('dt_overfitting_rmse.png', dpi=160, bbox_inches='tight'); plt.close()

def exp2_bagging_effect(X, y, seed=None, bags=20):
    Xtr, ytr, Xte, yte = split_train_test(X, y, seed=seed)
    leaf_sizes = [1,2,4,8,16,32,64,128]
    tr, te = [], []
    for ls in leaf_sizes:
        m = bl.BagLearner(learner=dt.DTLearner, kwargs={'leaf_size':ls}, bags=bags, boost=False)
        m.add_evidence(Xtr, ytr)
        tr.append(rmse(m.query(Xtr), ytr)); te.append(rmse(m.query(Xte), yte))
    plt.figure(); plt.plot(leaf_sizes, tr, marker='o', label=f'Train RMSE (bags={bags})')
    plt.plot(leaf_sizes, te, marker='o', label=f'Test RMSE (bags={bags})')
    plt.xscale('log', base=2); plt.xlabel('leaf_size (log2)'); plt.ylabel('RMSE')
    plt.title('Exp-2: Bagged DTLearner vs leaf_size'); plt.legend(); plt.grid(True, linewidth=0.3)
    plt.savefig('bagged_dt_rmse.png', dpi=160, bbox_inches='tight'); plt.close()

def exp3_dt_vs_rt(X, y, seed=None, leaf_size=8):
    Xtr, ytr, Xte, yte = split_train_test(X, y, seed=seed)
    d = dt.DTLearner(leaf_size=leaf_size); d.add_evidence(Xtr, ytr)
    r = rt.RTLearner(leaf_size=leaf_size); r.add_evidence(Xtr, ytr)
    dte, rte = d.query(Xte), r.query(Xte)
    labels = ['DT','RT']; mae_vals = [mae(dte,yte), mae(rte,yte)]; r2_vals = [r2(dte,yte), r2(rte,yte)]
    x = np.arange(len(labels)); w = 0.35; plt.figure()
    plt.bar(x-w/2, mae_vals, w, label='MAE'); plt.bar(x+w/2, r2_vals, w, label='R²'); plt.xticks(x, labels)
    plt.ylabel('Metric'); plt.title(f'Exp-3: DT vs RT (leaf_size={leaf_size})'); plt.legend(); plt.grid(True, axis='y', linewidth=0.3)
    plt.savefig('dt_vs_rt_metrics.png', dpi=160, bbox_inches='tight'); plt.close()
    return {'DT': {'MAE': mae_vals[0], 'R2': r2_vals[0]}, 'RT': {'MAE': mae_vals[1], 'R2': r2_vals[1]}}

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python testlearner.py Data/Istanbul.csv'); sys.exit(1)
    X, y = load_xy(sys.argv[1])
    seed = None
    t0 = time.time()
    exp1_dt_overfitting(X, y, seed=seed)
    exp2_bagging_effect(X, y, seed=seed)
    metrics = exp3_dt_vs_rt(X, y, seed=seed)
    with open('p3_results.txt', 'w', encoding='utf-8') as f:
        f.write('Experiment 3 – DT vs RT (test metrics)\n')
        for k in ['DT','RT']:
            f.write(f"{k}: MAE={metrics[k]['MAE']:.5f}, R2={metrics[k]['R2']:.5f}\n")
        f.write(f"Total runtime: {time.time()-t0:.2f}s\n")
    print('Saved: dt_overfitting_rmse.png, bagged_dt_rmse.png, dt_vs_rt_metrics.png, p3_results.txt')
