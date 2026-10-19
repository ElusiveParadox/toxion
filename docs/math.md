# Toxion: Mathematical reference

Notation: n molecules, T = 12 endpoints, labels y_it in {0, 1} or missing, parameters theta, gradient g = dL/dtheta, learning rate lr, step count t.

## 1. Endpoint dependence (`analysis/endpoints.py`)

For endpoints i and j, use only molecules labeled for both. Let n_ab count molecules with y_i = a, y_j = b (a, b in {0, 1}), and N = n00 + n01 + n10 + n11.

**phi coefficient.** The Pearson correlation of two binary variables simplifies to

    phi = (n11 n00 - n10 n01) / sqrt((n11 + n10)(n01 + n00)(n11 + n01)(n10 + n00))

phi is in [-1, 1]; it is undefined (NaN here) if either endpoint is constant on the shared molecules. phi^2 * N is approximately chi-squared with 1 degree of freedom under independence.

**Mutual information** (bits), with p(a, b) = n_ab / N and marginals p(a), p(b):

    I(A; B) = sum_{a,b} p(a, b) log2( p(a, b) / (p(a) p(b)) ),   with 0 log 0 = 0

I >= 0, I = 0 iff independent, and I(A; A) = H(A), the entropy. Unlike phi it also sees anti-correlation (a perfectly flipped endpoint has the same MI as an identical one). The plug-in estimate is biased upward when the pairwise sample is small.

Why it matters: strongly dependent endpoints are what multitask learning exploits; a correlation near 0 says sharing a representation helps little for that pair.

## 2. PCA via SVD (`analysis/chemspace.py`)

Let X be the n x d fingerprint matrix and Xc = X - mean(X). The SVD Xc = U S V^T gives:

- principal directions: the columns of V (rows of V^T);
- variance along component k: lambda_k = S_k^2 / (n - 1), the k-th eigenvalue of the sample covariance Xc^T Xc / (n - 1);
- explained variance ratio: lambda_k / sum_j lambda_j;
- scores (coordinates): Xc V_k = U_k S_k.

The scores are uncorrelated and each is centered. SVD is preferred to eigendecomposing the covariance because it avoids squaring the condition number. Signs of components are arbitrary; this code makes the largest-magnitude entry of each component positive.

Caveat: fingerprint bits are binary and sparse, so PCA variance is spread over many components; a low top-2 variance ratio is normal, not a bug.

## 3. Chemical-space measures

**Tanimoto similarity** of binary fingerprints a, b with |a| = number of set bits:

    T(a, b) = |a AND b| / (|a| + |b| - |a AND b|)

computed for all pairs as the matrix product A B^T (the intersections) plus row sums. Two empty fingerprints are given similarity 0.

**Nearest-neighbor similarity** of molecule i: max over j != i of T(i, j). A distribution concentrated near 1 means near-duplicate molecules, which is exactly what makes random splits optimistic.

**Internal diversity**: mean over distinct pairs of (1 - T), estimated on a random subsample of at most 2000 molecules.

**Scaffold entropy.** With scaffold frequencies p_s = (count_s) / n:

    H = - sum_s p_s log2 p_s,   effective number of scaffolds = 2^H

2^H is the number of equally common scaffolds that would give the same entropy. If the largest scaffold group is large (acyclic molecules form one group), a scaffold split puts that whole group in one side.

## 4. Optimizers (`optim/optimizers.py`)

All update rules follow torch.optim conventions and are verified numerically against it in the tests.

**SGD with momentum mu.** buf_1 = g_1; buf_t = mu buf_{t-1} + g_t; theta <- theta - lr buf_t. Momentum is an exponentially weighted sum of past gradients, which damps oscillation across steep directions. **Nesterov** uses the look-ahead direction g_t + mu buf_t.

**RMSprop.** v_t = alpha v_{t-1} + (1 - alpha) g_t^2; theta <- theta - lr g_t / (sqrt(v_t) + eps). Dividing by a running RMS of the gradient gives each parameter its own effective step size.

**Adam.** First and second moment estimates

    m_t = beta1 m_{t-1} + (1 - beta1) g_t
    v_t = beta2 v_{t-1} + (1 - beta2) g_t^2

start at 0, so they are biased toward 0 early. If g were constant, E[m_t] = (1 - beta1^t) g, so dividing by (1 - beta1^t) removes the bias (same for v with beta2):

    theta <- theta - lr * (m_t / (1 - beta1^t)) / (sqrt(v_t / (1 - beta2^t)) + eps)

On the first step this equals lr * g / (|g| + eps), about lr * sign(g), independent of gradient scale.

**Weight decay: L2 vs decoupled.** Adam with L2 adds lambda theta to the gradient, so the decay is then divided by sqrt(v): parameters with large gradient history are decayed less. **AdamW** instead shrinks the weights directly, theta <- theta (1 - lr lambda), before the Adam step, so the decay does not depend on the gradient history (Loshchilov and Hutter 2019).

## 5. Learning-rate schedules (`optim/schedules.py`)

lr_t = lr_base * f(t), t = 0, 1, ...

    warmup + cosine:  f(t) = (t + 1) / W                                       for t < W
                      f(t) = f_min + (1 - f_min) * 0.5 * (1 + cos(pi p)),  p = min(1, (t - W) / (T - W))   for t >= W

Warmup keeps Adam's early steps small while v_t is still a noisy estimate; cosine decay anneals smoothly to f_min. Step decay: f(t) = gamma^floor(t / s).

## 6. Losses for imbalanced multitask labels (`training/advanced_losses.py`)

Per entry with logit z, p = sigmoid(z), target y in {0, 1}: p_t = p if y = 1 else 1 - p, and BCE = -log p_t. Missing entries are masked out and every loss divides by the number of labeled entries.

**Class weighting.** Per endpoint, w = n_neg / n_pos (clipped to [0.1, 100]) multiplies the positive term:

    loss = -[ w y log p + (1 - y) log(1 - p) ]

At the optimum the model's probabilities are inflated for the positive class (the effective prior odds are multiplied by w), so weighted models are miscalibrated by construction; ranking metrics are not affected.

**Focal loss** (Lin et al. 2017): FL = alpha_t (1 - p_t)^gamma BCE, with alpha_t = alpha for positives and 1 - alpha for negatives. For a well-classified example with p_t = 0.98 and gamma = 2 the factor is 0.0004, so easy examples contribute almost nothing and the gradient focuses on hard ones. gamma = 0, alpha = None recovers BCE. Use gamma = 0 or gamma >= 1 (for 0 < gamma < 1 the gradient is unbounded as p_t -> 1).

**Uncertainty-weighted multitask loss** (Kendall et al. 2018). Treat each task's prediction as having homoscedastic noise sigma_t. With s_t = log sigma_t^2 learned jointly, the classification form used here is

    L = sum_t [ exp(-s_t) L_t + 0.5 s_t ]

For fixed L_t, dL/ds_t = -exp(-s_t) L_t + 0.5 = 0 gives s_t* = log(2 L_t), i.e. task weight 1 / (2 L_t): tasks with higher loss get lower weight, and the 0.5 s_t term stops the weights from collapsing to zero. This is the approximate form from the paper (an approximation of the softmax likelihood); treat it as a heuristic for classification, not an exact likelihood. Tasks with no labels in a batch are skipped.

## 7. Calibration (`eval/calibration.py`)

**ECE.** Split [0, 1] into B equal-width bins; with n_b predictions in bin b, observed frequency freq_b and mean predicted probability conf_b:

    ECE = sum_b (n_b / N) | freq_b - conf_b |

This is the positive-class version. It depends on B and is noisy for rare positives. **Brier score** = mean (p - y)^2.

**Temperature scaling.** Replace p by sigmoid(z / T), T > 0, fit on a validation set by minimizing the NLL. With beta = 1/T:

    NLL(beta) = sum_i [ softplus(beta z_i) - y_i beta z_i ]

This is a logistic loss in the single feature z, so it is convex in beta with a unique minimum (when both classes are present); T = 1/beta. Dividing logits by a positive constant preserves their order, so ROC-AUC and PR-AUC do not change. One temperature cannot fix a shifted base rate (as with class weighting); that needs a bias term (Platt scaling), which is not implemented here.
