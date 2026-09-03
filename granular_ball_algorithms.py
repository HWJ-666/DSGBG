# -*- coding: utf-8 -*-
"""DSGBG 及 6 种对比粒球生成算法的代码实现
- 每个 fit 返回 (centers, radii, labels, ball_sizes)，ball_sizes 为各粒球指派的样本数
算法清单：
  1. fit_dsgbg      DSGBG（本文方法）
  2. fit_scorgbg    ScOrGBC
  3. fit_ldgbg      LDGBG
  4. fit_origb      ORIGBG
  5. fit_accgbg     ACCGBG
  6. fit_adpgbg     ADPGBG
  7. fit_gbgpp      GBG++
预测器：gbknn_predict（GBKNN）/ gbknnpp_predict（GBKNN++）/ igbknn_predict（IGBKNN）
"""
import numpy as np
from collections import Counter

# ---------------------------------------------------------------------------
# 预测器
# ---------------------------------------------------------------------------

def gbknn_predict(X, centers, radii, ball_labels, ball_sizes=None):
    """GBKNN：测试样本分配给球面距离最近（D - r 最小）的粒球标签"""
    if len(centers) == 0:
        return np.zeros(len(X), dtype=int)
    D = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    return ball_labels[np.argmin(D - radii, axis=1)]

def gbknnpp_predict(X, centers, radii, ball_labels, ball_sizes):
    """GBKNN++：以球内样本数占比作为球面距离的权重"""
    if len(centers) == 0:
        return np.zeros(len(X), dtype=int)
    sw = ball_sizes / max(np.sum(ball_sizes), 1)
    D = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    return ball_labels[np.argmin(D - sw, axis=1)]

def igbknn_predict(X, centers, radii, ball_labels, ball_sizes):
    """IGBKNN：球内按多数类投票，球外按球面距离最近；球内冲突时取密度(样本数/半径)最大的球"""
    if len(centers) == 0:
        return np.zeros(len(X), dtype=int)
    density = ball_sizes / np.maximum(radii, 1e-8)
    D = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    surf = D - radii
    inside = surf <= 0
    yp = np.zeros(len(X), dtype=int)
    for i in range(len(X)):
        in_i = np.where(inside[i])[0]
        if len(in_i) == 0:
            yp[i] = ball_labels[np.argmin(surf[i])]
        elif len(in_i) == 1 or len(np.unique(ball_labels[in_i])) == 1:
            yp[i] = ball_labels[in_i[0]]
        else:
            yp[i] = ball_labels[in_i[np.argmax(density[in_i])]]
    return yp

# ---------------------------------------------------------------------------
# 1. DSGBG：基于双尺度截断距离与类间分离度感知的粒球生成（本文方法）
# ---------------------------------------------------------------------------

def fit_dsgbg(Xtr, ytr, sig_coef=5.0, pct_slope=15.0):
    """DSGBG返回 (centers, radii, labels, ball_sizes)。
    sig_coef / pct_slope 为设计阶段确定的固定常数。
    """
    X = Xtr.astype(np.float64)
    y = ytr.astype(np.int64)
    cl = np.unique(y)

    # ---- Phase 0: 全局类间分离度 λ ----
    intra = []
    for c in cl:
        Xc = X[y == c]
        if len(Xc) > 1:
            Dc = np.linalg.norm(Xc[:, None, :] - Xc[None, :, :], axis=2)
            intra.append(Dc[np.triu_indices(len(Xc), k=1)].mean())   # 类内样本对距离的均值
    intra_d = np.mean(intra) if intra else 0.0

    inter = []
    for i in range(len(cl)):
        for j in range(i + 1, len(cl)):
            X1 = X[y == cl[i]]
            X2 = X[y == cl[j]]
            if len(X1) and len(X2):
                D12 = np.linalg.norm(X1[:, None, :] - X2[None, :, :], axis=2)
                inter.append(D12.mean())
    inter_d = np.mean(inter) if inter else 0.0
    lamb = 0.5 if inter_d == 0 else float(
        1.0 / (1.0 + np.exp(-sig_coef * (inter_d - intra_d) / inter_d)))

    all_c, al, all_r, all_scores = [], [], [], []
    suppressed_pts = []          # (全局索引, 类, dc, 最近异类距离, 最近同类距离, 评分)

    for c in cl:
        ig = np.where(y == c)[0]
        Xc = X[ig]
        Nc = len(ig)
        Xo = X[y != c]

        if Nc < 2:
            all_c.append(Xc.mean(axis=0))
            al.append(c)
            all_r.append(1.0)
            all_scores.append(0.0)
            continue

        # ---- 双尺度截断距离 ----
        Dc = np.linalg.norm(Xc[:, None, :] - Xc[None, :, :], axis=2)
        knn = np.sort(Dc, axis=1)
        k = max(1, int(0.015 * Nc))                        # 邻居数 K = 0.015·Nc
        dc_local = max(float(np.median(knn[:, k])), 1e-8)  # 第K近邻距离中位数

        tri = np.triu_indices(Nc, k=1)
        all_d = Dc[tri]                                    # 类内成对距离集合
        cv_val = float(np.std(all_d) / max(np.mean(all_d), 1e-8))   # 变异系数
        pct = 5 + pct_slope * cv_val                       # 自适应百分位
        dc_global = float(np.percentile(all_d, pct))       # 全局尺度

        alpha = lamb / (1.0 + cv_val)                      # 混合系数
        dc = alpha * dc_local + (1 - alpha) * dc_global    # 最终截断距离

        # ---- 局部密度：邻居计数 + 高斯核加权 ----
        in_dc = Dc <= dc
        dens = in_dc.sum(axis=1).astype(np.float64)  # 与LDGBG一致计算
        for i in range(Nc):
            nb = np.where(in_dc[i])[0]
            nb = nb[nb != i]
            if len(nb) > 0:
                dens[i] += np.sum(np.exp(-(Dc[i][nb] ** 2) / (dc ** 2)))
        rho_m = dens.max()
        rho_n = dens / rho_m if rho_m > 0 else np.zeros(Nc)

        # ---- 边界距离：到最近异类样本的欧氏距离 ----
        if len(Xo) > 0:
            Do = np.linalg.norm(Xo[:, None, :] - Xc[None, :, :], axis=2)
            hdist_1 = Do.min(axis=0)
            mg = np.maximum(0, hdist_1 - 1e-6)
        else:
            hdist_1 = np.full(Nc, np.inf)
            mg = np.full(Nc, 1e10)
        mg_max = mg.max()
        mg_n = mg / mg_max if mg_max > 0 else np.zeros(Nc)

        # ---- 球心评分：融合归一化后的边界距离与局部密度 ----
        scores = lamb * mg_n + (1 - lamb) * rho_n
        r_star = np.maximum(np.minimum(dc, hdist_1), 1e-8)

        # 离群点抑制：类内是否存在"邻域内不止自身"的样本
        has_noniso = bool(np.any((Dc <= r_star).sum(axis=1) >= 2))

        # ---- 贪心覆盖选心 ----
        order = np.argsort(scores)[::-1]
        remaining = set(range(Nc))
        for oi in order:
            if oi not in remaining:
                continue
            if has_noniso:
                nb0 = np.where(Dc[oi] <= r_star[oi])[0]
                if len(nb0) <= 1:      # 邻域内仅含自身：标记为离群点候选
                    knn_i = np.sort(Dc[oi])
                    d_same = float(knn_i[1]) if Nc > 1 else np.inf
                    suppressed_pts.append((ig[oi], c, dc, float(hdist_1[oi]),
                                           d_same, float(scores[oi])))
                    remaining.discard(oi)
                    continue
            already = any(al[j] == c and np.linalg.norm(all_c[j] - Xc[oi]) <= dc
                          for j in range(len(all_c)))
            if already:
                remaining.discard(oi)
                continue
            nbi = np.where(Dc[oi] <= dc)[0]
            r_calib = float(np.max(Dc[oi][nbi])) if len(nbi) > 1 else dc  # 校准半径
            all_c.append(Xc[oi])
            al.append(c)
            all_r.append(float(min(r_calib, hdist_1[oi])))   # 半径取校准半径与最近异类距离的较小值
            all_scores.append(scores[oi])
            remaining.difference_update(nbi)

    if not all_c:
        return (np.array([]).reshape(0, X.shape[1]), np.array([]),
                np.array([], dtype=y.dtype), np.array([]))

    # ---- 全局 kNN（供纠错补选与混合投票标签）----
    k_nb = max(1, int(0.015 * len(X))) #与上述k近邻的k一致
    k_nb = min(k_nb, len(X) - 1)
    DX = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(DX, np.inf)
    knn_idx = np.argsort(DX, axis=1)[:, :k_nb]

    def assign_and_label(centers, radii, ca):
        K = len(centers)
        D = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        ass = np.argmin(D - radii, axis=1)
        bl = np.array([Counter(y[ass == j]).most_common(1)[0][0]
                       if np.sum(ass == j) > 0 else ca[j] for j in range(K)])
        return ass, bl

    centers = np.vstack(all_c)
    radii = np.array([max(r, 1e-8) for r in all_r])
    ca = np.array(al)
    ass0, bl0 = assign_and_label(centers, radii, ca)

    # ---- 离群点纠错补选：被抑制点中"删除会错且为结构点"者恢复为候选球心 ----
    for (gidx, c, dc_c, hdist_o, d_same_o, score_o) in suppressed_pts:
        d_own = np.linalg.norm(X[gidx] - centers, axis=1)
        own_ball = int(np.argmin(d_own - radii))
        if bl0[own_ball] == y[gidx]:         # 判据1：删除不会造成错分 → 不补
            continue
        if hdist_o < 0.5 * d_same_o - 1e-8:  # 判据3：异类间隙过小（标签孤立/噪声）→ 不补
            continue
        nbs = knn_idx[gidx]
        if len(nbs) == 0:
            continue
        if Counter(y[nbs]).most_common(1)[0][0] != y[gidx]:  # 判据2：非结构点 → 不补
            continue
        all_c.append(X[gidx])
        al.append(c)
        all_r.append(float(min(dc_c, hdist_o)))
        all_scores.append(score_o)

    centers = np.vstack(all_c)
    radii = np.array([max(r, 1e-8) for r in all_r])
    ca = np.array(al)

    # ---- 最终指派 + 混合投票标签（球内多数票 + 球心全局kNN多数票）----
    ass, _ = assign_and_label(centers, radii, ca)
    sizes = np.array([np.sum(ass == j) for j in range(len(centers))])
    DC_cent = np.linalg.norm(centers[:, None, :] - X[None, :, :], axis=2)
    cent_knn = np.argsort(DC_cent, axis=1)[:, :min(k_nb + 1, len(X))]
    bl = np.zeros(len(centers), dtype=y.dtype)
    for j in range(len(centers)):
        votes = Counter(y[ass == j])
        nbs = cent_knn[j][1:] if len(cent_knn[j]) > 1 else cent_knn[j]  # 去掉球心自身
        votes.update(y[nbs])
        bl[j] = votes.most_common(1)[0][0]

    return centers, radii, bl, sizes

# ---------------------------------------------------------------------------
# 2. ScOrGBC：基于 K-means 的稳定中心与最优半径
# ---------------------------------------------------------------------------

def _kmeans_pp(X_sc, k):
    """K-means++ 概率初始化"""
    n = X_sc.shape[0]
    cents = [X_sc[np.random.choice(n)].copy()]
    min_D2 = np.sum((X_sc - cents[0]) ** 2, axis=1)
    for _ in range(1, k):
        s = min_D2.sum()
        probs = np.ones(n) / n if s < 1e-12 else min_D2 / s
        new_c = X_sc[np.random.choice(n, p=probs)].copy()
        cents.append(new_c)
        d2_new = np.sum((X_sc - new_c) ** 2, axis=1)
        min_D2 = np.minimum(min_D2, d2_new)
    return np.array(cents)

def _kmeans_cluster(X_sc, cents, max_iter=10):
    """K-means 迭代至中心收敛"""
    for _ in range(max_iter):
        D2 = np.sum(X_sc ** 2, axis=1)[:, None] + np.sum(cents ** 2, axis=1)[None, :] \
             - 2 * X_sc @ cents.T
        labels_k = np.argmin(D2, axis=1)
        new_c = np.array([X_sc[labels_k == k].mean(0) if np.any(labels_k == k)
                          else X_sc[np.argmin(D2[:, k])] for k in range(len(cents))])
        if np.allclose(cents, new_c):
            cents = new_c
            break
        cents = new_c
    return cents, labels_k

def _eliminate_overlap(GBS, epsilon=1e-6, max_iter=200):
    """异类粒球重叠压缩：压缩量按半径正比分配"""
    N = len(GBS)
    if N <= 1:
        return GBS
    radii_ol = [g[1] for g in GBS]
    for _ in range(max_iter):
        delta = np.zeros(N)
        for i in range(N):
            max_od, bj = -1.0, -1
            for j in range(N):
                if i == j or GBS[i][2] == GBS[j][2]:
                    continue
                od = radii_ol[i] + radii_ol[j] - np.linalg.norm(GBS[i][0] - GBS[j][0])
                if od > max_od:
                    max_od, bj = od, j
            if bj != -1 and max_od > 0:
                delta[i] = max_od * radii_ol[i] / (radii_ol[i] + radii_ol[bj])
        if np.sum(delta) < epsilon:
            break
        for i in range(N):
            radii_ol[i] = max(0, radii_ol[i] - delta[i])
    for i in range(N):
        GBS[i][1] = radii_ol[i]
    return GBS

def _optimize_radii(GBS, X_sc, y_sc, labels_km, beta=1.0, n_candidates=21):
    """合理粒度原则优化半径：最大化 覆盖率×exp(-β·r)"""
    for i in range(len(GBS)):
        center = GBS[i][0]
        r0 = GBS[i][1]
        if r0 < 1e-6:
            continue
        cluster_mask = labels_km == i
        if cluster_mask.sum() == 0:
            continue
        X_cluster = X_sc[cluster_mask]
        dists = np.linalg.norm(X_cluster - center, axis=1)
        candidates = np.linspace(r0 / 2, r0, n_candidates)
        best_r, best_j = r0, -np.inf
        for r in candidates:
            coverage_val = np.sum(dists <= r)
            j = coverage_val * np.exp(-beta * r)
            if j > best_j:
                best_j, best_r = j, r
        GBS[i][1] = best_r
    return GBS

def fit_scorgbg(Xtr, ytr, t=1.0, beta=1.0):
    """ScOrGBC。t 为球数缩放系数，β 为粒度权重（网格搜索确定）"""
    X = np.asarray(Xtr)
    y = np.asarray(ytr)
    n_samples = X.shape[0]
    Ni = max(1, int(np.floor(t * np.sqrt(n_samples))))    # 球数 K = t·√N
    cents = _kmeans_pp(X, Ni)
    cents, labels_km = _kmeans_cluster(X, cents)
    GBS = []
    for k in range(Ni):
        mask = labels_km == k
        Xk = X[mask]
        if len(Xk) == 0:
            GBS.append([cents[k], 0.0, -1])
        else:
            r = np.max(np.linalg.norm(Xk - cents[k], axis=1))
            lbl = Counter(y[mask]).most_common(1)[0][0]
            GBS.append([cents[k], r, lbl])
    GBS = _eliminate_overlap(GBS)
    GBS = _optimize_radii(GBS, X, y, labels_km, beta=beta)
    sizes = np.array([np.sum(labels_km == k) for k in range(Ni)])
    return (np.array([g[0] for g in GBS]), np.array([g[1] for g in GBS]),
            np.array([g[2] for g in GBS]), sizes)

# ---------------------------------------------------------------------------
# 3. LDGBG：基于局部密度的粒球生成
# ---------------------------------------------------------------------------

def fit_ldgbg(Xtr, ytr, g=1.0):
    """LDGBG。g 为粒度系数（逐数据集搜索），邻域半径 δ = g·dc"""
    X = Xtr.astype(np.float64)
    y = ytr.astype(np.int64)
    cl = np.unique(y)
    sparsity = len(X) / max(X.shape[1], 1)
    is_sparse = sparsity < 40          # LDGBG 的稀疏度阈值

    all_c, al, ad, all_dens = [], [], [], []

    for c in cl:
        ig = np.where(y == c)[0]
        Xc = X[ig]
        Nc = len(ig)
        if Nc < 2:
            all_c.append(Xc.mean(axis=0))
            al.append(c)
            ad.append(1.0)
            all_dens.append(0.0)
            continue

        # dc 取第 K 近邻距离中位数（K = 0.015·Nc），邻域半径 δ = g·dc
        Dc = np.linalg.norm(Xc[:, None, :] - Xc[None, :, :], axis=2)
        knn = np.sort(Dc, axis=1)
        k = max(1, int(0.015 * Nc))
        dc = max(float(np.median(knn[:, k])), 1e-8)
        radius = max(g * dc, 1e-8)

        # 局部密度：邻居计数 + 高斯核加权
        in_r = Dc <= radius
        cnt = in_r.sum(axis=1)
        dens = cnt.astype(np.float64)
        for i in range(Nc):
            nb = np.where(in_r[i])[0]
            nb = nb[nb != i]
            if len(nb) > 0:
                dens[i] += np.sum(np.exp(-(Dc[i][nb] ** 2) / (radius ** 2)))
        rho_max = dens.max()

        # 按密度降序贪心选心，覆盖半径内的样本密度清零
        while True:
            oi = np.argmax(dens)
            if dens[oi] < 0.01 * rho_max:
                break
            if not is_sparse and cnt[oi] <= 1:
                dens[oi] = 0
                continue
            all_c.append(Xc[oi])
            al.append(c)
            ad.append(radius)
            all_dens.append(dens[oi])
            within = np.where(in_r[oi])[0]
            dens[within] = 0

    if not all_c:
        return (np.array([]).reshape(0, X.shape[1]), np.array([]),
                np.array([], dtype=y.dtype), np.array([]))

    centers = np.vstack(all_c)
    ca = np.array(al)
    ball_dens = np.array(all_dens)
    K = len(centers)

    # 半径：同类取球心距一半；异类按密度比缩短
    cdist_all = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    radii = np.zeros(K)
    for i in range(K):
        min_d, nj = np.inf, -1
        for j in range(K):
            if i == j:
                continue
            if cdist_all[i, j] < min_d:
                min_d, nj = cdist_all[i, j], j
        if nj == -1:
            radii[i] = float(ad[i])
        elif ca[i] == ca[nj]:
            radii[i] = min_d / 2
        else:
            rho_i, rho_j = ball_dens[i], ball_dens[nj]
            if rho_i > rho_j:
                radii[i] = min_d / 2
            else:
                radii[i] = min_d / 2 * rho_i / (rho_i + rho_j)
        radii[i] = max(radii[i], 0.0)

    # 指派与标签
    D = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    ass = np.argmin(D - radii, axis=1)
    bl = np.array([Counter(y[ass == j]).most_common(1)[0][0]
                   if np.sum(ass == j) > 0 else ca[j] for j in range(K)])
    sizes = np.array([np.sum(ass == j) for j in range(K)])
    return centers, radii, bl, sizes

# ---------------------------------------------------------------------------
# 4. ORIGBG：纯度阈值驱动的 2-means 递归分裂
# ---------------------------------------------------------------------------

def fit_origb(Xtr, ytr, T=1.0):
    """ORIGBG。T 为纯度阈值（取 1.0）"""
    from sklearn.cluster import KMeans
    X = np.asarray(Xtr, dtype=np.float64)
    y = np.asarray(ytr, dtype=np.int64)
    n = len(X)

    def _purity(lb):
        if len(lb) == 0:
            return 0.0
        return Counter(lb).most_common(1)[0][1] / len(lb)

    def _label(lb):
        return Counter(lb).most_common(1)[0][0]

    balls = [np.arange(n)]
    changed = True
    it = 0
    while changed and it < 1000:
        it += 1
        changed = False
        nb = []
        for idx in balls:
            if len(idx) < 2:
                nb.append(idx)
                continue
            yb = y[idx]
            Xb = X[idx]
            if _purity(yb) >= T or len(np.unique(yb)) < 2:
                nb.append(idx)
                continue
            if np.unique(Xb, axis=0).shape[0] < 2:   
                nb.append(idx)
                continue
            km = KMeans(n_clusters=2, n_init=1, init='random', random_state=42)
            lbs = km.fit_predict(Xb)
            for ci in range(2):
                nb.append(idx[lbs == ci])
            changed = True
        if len(nb) == len(balls):
            break
        balls = nb

    result = [(X[idx].mean(0),
               max(np.mean(np.linalg.norm(X[idx] - X[idx].mean(0), axis=1)), 0.01),
               _label(y[idx])) for idx in balls if len(idx) > 0]
    if not result:
        return np.zeros((0, X.shape[1])), np.zeros(0), np.zeros(0, int), np.zeros(0)
    sizes = np.array([len(idx) for idx in balls if len(idx) > 0])
    return (np.array([r[0] for r in result]), np.array([r[1] for r in result]),
            np.array([r[2] for r in result]), sizes)

# ---------------------------------------------------------------------------
# 5. ACCGBG：K-division 加速 + 全局划分
# ---------------------------------------------------------------------------

def fit_accgbg(Xtr, ytr, T=1.0):
    """ACCGBG。T 为纯度阈值"""
    from sklearn.cluster import KMeans
    X = np.asarray(Xtr, dtype=np.float64)
    y = np.asarray(ytr, dtype=np.int64)
    data = np.hstack([y.reshape(-1, 1), X])

    def _gb_label(gb):
        return Counter(gb[:, 0]).most_common(1)[0][0]

    def _gb_purity(gb):
        if len(gb) == 0:
            return 0.0
        return max(np.sum(gb[:, 0] == l) for l in np.unique(gb[:, 0])) / len(gb)

    def _ctr_rad(gb):
        c = gb[:, 1:].mean(0)
        r = np.mean(np.sqrt(np.sum((gb[:, 1:] - c) ** 2, axis=1)))
        return c, max(r, 0.01)

    # K-division：父球均值中心 + k-1 个异类随机点，一次分配完成分裂
    gb_list = [data]
    changed = True
    it = 0
    while changed and it < 1000:
        it += 1
        changed = False
        nb = []
        for gb in gb_list:
            if len(gb) < 2:
                nb.append(gb)
                continue
            ula = np.unique(gb[:, 0])
            k = len(ula)
            if k < 2:
                nb.append(gb)
                continue
            if _gb_purity(gb) >= T:
                nb.append(gb)
                continue
            pc, _ = _ctr_rad(gb)
            centers = [pc]
            for ol in [l for l in ula.tolist() if l != _gb_label(gb)]:
                ogb = gb[gb[:, 0] == ol]
                if len(ogb) > 0:
                    centers.append(ogb[np.random.randint(0, len(ogb)), 1:])
            centers = np.array(centers)
            dists = np.linalg.norm(gb[:, 1:][:, None, :] - centers[None, :, :], axis=2)
            lbs = np.argmin(dists, axis=1)
            for ci in range(k):
                nb.append(gb[lbs == ci])
            changed = True
        if len(nb) == len(gb_list):
            break
        gb_list = nb

    result = []
    for gb in gb_list:
        if len(gb) >= 2:
            c, r = _ctr_rad(gb)
            l = _gb_label(gb)
            result.append((c, r, l))
    if len(result) <= 1:
        if not result:
            return np.zeros((0, X.shape[1])), np.zeros(0), np.zeros(0, int), np.zeros(0)
        return (np.array([r[0] for r in result]), np.array([r[1] for r in result]),
                np.array([r[2] for r in result]),
                np.array([len(gb) for gb in gb_list if len(gb) >= 2]))

    # 全局划分：以分裂球心为初始中心做 K-means
    centers = np.array([r[0] for r in result])
    if np.unique(data[:, 1:], axis=0).shape[0] < len(centers):
        # 不同坐标点不足：K-means 无法运行，直接返回分裂结果
        sizes = np.array([len(gb) for gb in gb_list if len(gb) >= 2])
        return (np.array([r[0] for r in result]), np.array([r[1] for r in result]),
                np.array([r[2] for r in result]), sizes)
    km = KMeans(n_clusters=len(centers), n_init=2, init=centers, random_state=5)
    lbs = km.fit_predict(data[:, 1:])
    final = []
    for ci in range(len(centers)):
        sub = data[lbs == ci]
        if len(sub) >= 2:
            c, r = _ctr_rad(sub)
            l = _gb_label(sub)
            final.append((c, r, l))
    if not final:
        final = result
    sizes = np.array([len(data[lbs == ci]) for ci in range(len(centers))])
    return (np.array([f[0] for f in final]), np.array([f[1] for f in final]),
            np.array([f[2] for f in final]), sizes[:len(final)])

# ---------------------------------------------------------------------------
# 6. ADPGBG：基于最短异类距离的自适应粒球生成
# ---------------------------------------------------------------------------

def _center_radius(Xs):
    if len(Xs) == 0:
        return None, 0.0
    c = np.mean(Xs, axis=0)
    r = np.mean(np.linalg.norm(Xs - c, axis=1))
    return c, r if r > 0 else 0.1

def fit_adpgbg(Xtr, ytr):
    """ADPGBG。以最近异类距离为半径构建纯球，零参数"""
    X = Xtr.astype(np.float64)
    y = ytr.astype(np.int64)
    N = len(X)

    def hd(ci):
        """第一、第二最短异类距离"""
        xc = X[ci]
        lc = y[ci]
        d = np.linalg.norm(X - xc, axis=1)
        hdists = np.sort(d[y != lc])
        if len(hdists) == 0:
            return np.inf, np.inf
        if len(hdists) == 1:
            return hdists[0], np.inf
        return hdists[0], hdists[1]

    Ds = set(range(N))
    bi = []

    while len(Ds) > 0:
        dl = list(Ds)
        yD = y[dl]
        uc = np.unique(yD)
        if len(uc) == 0:
            break

        # 每类随机选一个候选中心
        tc = []
        for cls in uc:
            cand = [i for i in Ds if y[i] == cls]
            if cand:
                tc.append(np.random.choice(cand))

        for ci in tc:
            if ci not in Ds:
                continue
            h1, h2 = hd(ci)
            if h1 == np.inf:
                continue

            # 最近异类距离范围内的样本构成纯粒球
            dci = np.linalg.norm(X[dl] - X[ci], axis=1)
            Ti = [dl[i] for i in np.where(dci < h1 - 1e-12)[0]]
            if len(Ti) == 0:
                Ti = [ci]

            if len(Ti) == 1:
                # 球内仅含自身：检查第二最短异类距离范围内的同类样本
                if h2 == np.inf:
                    Ds.discard(Ti[0])
                    continue
                dw = np.where((np.linalg.norm(X - X[Ti[0]], axis=1) <= h2 + 1e-12)
                              & (y == y[Ti[0]]))[0]
                if len(dw) <= 1:      # 同类样本不足：判定为离群点并剔除
                    Ds.discard(Ti[0])
                    continue
                cc, cr = _center_radius(X[dw])
                bi.append({'indices': list(dw), 'center': cc, 'radius': cr, 'label': y[Ti[0]]})
                Ds.difference_update(dw)
            else:
                cc, cr = _center_radius(X[Ti])
                bi.append({'indices': Ti, 'center': cc, 'radius': cr, 'label': y[ci]})
                Ds.difference_update(Ti)

    # 局部优化：重新计算中心与半径，标签取多数类
    for b in bi:
        if len(b['indices']) > 0:
            b['center'], b['radius'] = _center_radius(X[b['indices']])
            b['label'] = Counter(y[b['indices']]).most_common(1)[0][0]

    if not bi:
        return np.array([]).reshape(0, X.shape[1]), np.array([]), np.array([], dtype=y.dtype), np.array([])

    sizes = np.array([len(b['indices']) for b in bi])
    return (np.array([b['center'] for b in bi]),
            np.array([b['radius'] for b in bi]),
            np.array([b['label'] for b in bi]),
            sizes)

# ---------------------------------------------------------------------------
# 7. GBG++：基于注意力机制的快速稳定粒球生成方法
# ---------------------------------------------------------------------------

def fit_gbgpp(Xtr, ytr):
    """GBG++。数据驱动中心 + 离群点剔除 + 冲突消解"""
    X = Xtr.astype(np.float64)
    y = ytr.astype(np.int64)

    def _ctr_rad(pts):
        c = pts.mean(0)
        r = np.mean(np.linalg.norm(pts - c, axis=1))
        return c, max(r, 0.01)

    def _lbl_pur(idx):
        if len(idx) == 0:
            return -1, 0.0
        cnt = Counter(y[idx])
        lbl = cnt.most_common(1)[0][0]
        return lbl, cnt[lbl] / len(idx)

    class GB:
        def __init__(s, idx):
            s.i = np.array(idx)
            s.c, s.r = _ctr_rad(X[s.i])
            s.l, s.p = _lbl_pur(s.i)

    def split_gb(gb):
        idx = gb.i.copy()
        rem = idx.copy()
        children = []
        while len(rem) > len(np.unique(y[rem])):
            yt = y[rem]
            maj = Counter(yt).most_common(1)[0][0]
            midx = rem[yt == maj]
            if len(midx) < 2:
                break
            cc = X[midx].mean(0)
            rc = np.mean(np.linalg.norm(X[midx] - cc, axis=1))
            mask = np.linalg.norm(X[rem] - cc, axis=1) <= rc
            if mask.sum() == 0:
                break
            child = GB(rem[mask])
            if len(child.i) > 1:
                children.append(child)
            rem = rem[~mask]
            if len(rem) > 0 and len(np.unique(y[rem])) == 1:
                last = GB(rem)
                if len(last.i) > 1:
                    children.append(last)
                break
        # 冲突消解：合并异类嵌套粒球后重新分裂
        if len(children) > 1:
            merged = True
            while merged:
                merged = False
                for i in range(len(children)):
                    for j in range(i + 1, len(children)):
                        a, b = children[i], children[j]
                        if a.l != b.l and np.linalg.norm(a.c - b.c) <= abs(a.r - b.r):
                            mg = GB(np.concatenate([a.i, b.i]))
                            children = [children[k] for k in range(len(children))
                                        if k not in (i, j)]
                            children.append(mg)
                            merged = True
                            break
                    if merged:
                        break
        return children

    gb_list = [GB(np.arange(len(X)))]
    i = 0
    n = len(gb_list)
    while True:
        if gb_list[i].p < 1.0:
            children = split_gb(gb_list[i])
            if len(children) > 1:
                gb_list[i] = children[0]
                gb_list.extend(children[1:])
            elif len(children) == 1 and len(children[0].i) != len(gb_list[i].i):
                gb_list.pop(i)
                gb_list.append(children[0])
            else:
                gb_list.pop(i)
            n = len(gb_list)
        else:
            i += 1
        if i == n:
            break

    if not gb_list:
        return np.zeros((0, X.shape[1])), np.zeros(0), np.zeros(0, int), np.zeros(0)
    sizes = np.array([len(g.i) for g in gb_list])
    return (np.array([g.c for g in gb_list]), np.array([g.r for g in gb_list]),
            np.array([g.l for g in gb_list]), sizes)
