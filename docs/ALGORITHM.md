# The algorithm

Two models, trained separately and blended at request time. This page is the
maths, and — more usefully — the reasoning behind the parts of it that are easy
to get subtly wrong.

Notation follows Andrew Ng's formulation throughout:

| symbol | shape | meaning |
| --- | --- | --- |
| `n_u`, `n_m`, `n` | | users, menu items, latent features (10) |
| `Y` | `(n_m, n_u)` | ratings, 1–5 |
| `R` | `(n_m, n_u)` | 1 where user `j` rated item `i`, else 0 |
| `X` | `(n_m, n)` | learned item features |
| `W` | `(n_u, n)` | learned user parameters |
| `b` | `(1, n_u)` | per-user bias |
| `mu` | `(n_m,)` | per-item mean rating |

---

## Phase A — Collaborative filtering

### The objective

Prediction for one pair:

```
prediction(i, j) = W[j] · X[i] + b[0, j]
```

`X`, `W` and `b` are all unknown and learned **simultaneously** by minimising

```
J(W, b, X) = ½ · Σ         (W[j]·X[i] + b[0,j] − Y[i,j])²
                 (i,j): R[i,j]=1

           + (λ/2) · Σ W²  +  (λ/2) · Σ X²
```

Three things about this cost are worth stating explicitly.

**The sum runs only over observed entries.** `R` is a mask, not data. An unrated
pair is *unknown*, not a zero — a dish nobody has ordered is not a dish everyone
hated. The implementation applies the mask to the error rather than selecting
entries, so the whole thing stays one dense matrix operation that `GradientTape`
can differentiate:

```python
errors = (X @ W.T + b - Y) * R          # unrated terms become exactly 0
J = 0.5 * sum(errors ** 2) + (λ/2) * (sum(X**2) + sum(W**2))
```

There is a test in `tests/test_cofi.py` that tampers with every masked entry,
setting it to 999, and asserts the cost does not move.

**`b` is not regularised.** The penalty covers `W` and `X` only. Shrinking the
bias toward zero would fight the per-user baseline it exists to represent —
some people simply rate everything a 4, and that is signal, not overfitting.

**`W` and `b` stay separate tensors.** They are often folded into one augmented
matrix by appending a column of ones to `X`. That trick is tidy and it silently
subjects the bias to `W`'s regularisation, which is the previous point again.

### Optimisation

A hand-written loop over `tf.GradientTape`, not `model.fit`. There is no Keras
model to fit: `X`, `W` and `b` are free parameters optimised jointly against a
custom objective, not layers of a network.

```python
for iteration in range(iterations):
    with tf.GradientTape() as tape:
        cost = cofi_cost_func(X, W, b, Y, R, lambda_)
    gradients = tape.gradient(cost, [X, W, b])
    optimizer.apply_gradients(zip(gradients, [X, W, b]))
```

Adam at `learning_rate=1e-1`. Cost falls from 5,691 to 1,388 in 400 iterations
and is flat by roughly 200 — see the curve in
[`EVALUATION.md`](EVALUATION.md#training).

Parameters are initialised at `0.1 × N(0, 1)`. With standard-normal init and
n=10, predictions start with a standard deviation around √10, which swamps
mean-centred targets living in roughly [−2, 2]; the first few hundred steps then
go on recovering from the starting point rather than learning anything.

### Mean normalisation, and why it is not optional

Before training, each item's mean over its *rated* entries is subtracted:

```
mu[i]        = mean of Y[i, j] over j where R[i,j] = 1
Y_norm[i, j] = (Y[i, j] − mu[i]) · R[i, j]
```

and added back at prediction time:

```
prediction(i, j) = W[j] · X[i] + b[0, j] + mu[i]
```

**Consider a user who has rated nothing.** No term in the cost function involves
their column, so the only force acting on `W[j]` is the regulariser, which pulls
it to the origin. `b[0,j]` starts at zero and stays there. Their prediction for
every dish on the menu is therefore

```
0 · X[i] + 0 = 0
```

Zero is not a neutral rating on a 1–5 scale. It is worse than the worst review
anyone could leave, and it is *identical for every item*, so the entire menu
ranks equally badly and the ordering is arbitrary.

With normalisation the same user gets `0 + mu[i]` — each dish's average rating.
That is exactly the right thing to believe about somebody you know nothing
about, and it produces a sensible ordering immediately.

This is verified against a real trained model, not just asserted: for a seeded
user with zero ratings, `|W[j]|` converges to 2.7e-10 and `b[0,j]` to exactly 0,
so their predictions match the item means to within 3.4e-10 and land in
2.40–5.00 rather than at zero.

One extension beyond the standard formulation: **an item nobody has rated has no
mean of its own**, so `mu[i]` falls back to the global mean rather than zero.
Leaving it at zero would sink every new dish to the bottom of every ranking for
reasons unrelated to its quality.

### Related items

After training, similarity between dishes is squared distance in the learned
feature space:

```
distance(k, i) = ‖ X[k] − X[i] ‖²
```

computed as `‖x_k‖² + ‖x_i‖² − 2·x_k·x_i` so the whole matrix is one multiply,
and the top 10 per item are written to `item_neighbors`. It is precomputed
because the answer only changes when the model is retrained, and doing it per
request would put an O(n_m) pass on the hot path for a fixed result.

Measured quality, against a 15.7% random baseline: **47.8%** of an item's top-10
neighbours share its cuisine.

### Implicit feedback

Restaurants have order logs, not star ratings. The same machinery takes binary
labels — `1` shown and ordered, `0` shown and not ordered, and *nothing* for
never shown — with a sigmoid and cross-entropy instead of squared error:

```
prediction(i, j) = g(W[j] · X[i] + b[0, j])
J = Σ  −[y·log g(z) + (1−y)·log(1−g(z))]  +  regularisation
    R=1
```

Two notes. The implementation uses `sigmoid_cross_entropy_with_logits` rather
than taking the log of an explicit sigmoid, which overflows the moment a logit
saturates. And **mean normalisation is not applied on this path**: subtracting an
average from a 0/1 label and pushing the result through a sigmoid is not a
defined operation. Item popularity is absorbed by the bias term instead.

`R` matters more here than in the explicit case. Treating every non-order as a
rejection would teach the model that customers dislike every dish they were
never shown. This is why the `impressions` table exists — without logged
non-conversions there is only `y=1` data, and no model can learn what people
turn down.

---

## Why collaborative filtering alone is not enough

It fails in exactly two places, and both are routine rather than exotic.

**A new customer.** Their column of `Y` is empty, so — as above — `W[j]` is
whatever the regulariser leaves and the model has no opinion about them. Mean
normalisation makes the failure graceful, but the result is still "here is what
is popular", which is not personalisation.

**A new dish.** Its row of `Y` is empty, so `X[i]` is pulled to the origin. It
then sits near every other unplaced dish in feature space, which makes its
nearest neighbours an artifact of regularisation rather than a real similarity.
`build_neighbors` counts and reports these rather than writing them out as
though they meant something.

Both failures share a cause: **collaborative filtering only knows ids**. It
learns from the pattern of who rated what and has no access to the fact that a
dish is a spicy mutton biryani costing 520 taka, or that the customer said at
signup that they can handle any amount of chilli. That information exists. The
second model uses it.

---

## Phase B — Content-based two-tower model

Two independent networks embed a user and a dish into the same space:

```
v_u = l2_normalize(user_NN(x_u))        user_NN: 256 → 128 → 32
v_m = l2_normalize(item_NN(x_m))        item_NN: 256 → 128 → 32
prediction = v_u · v_m
```

The input vectors are different lengths — 35 user features against 55 item
features — and the towers project both to 32 dimensions, which is the entire
point of the architecture.

| `x_u` | `x_m` |
| --- | --- |
| age bucket, gender, area | cuisine, spice level |
| total orders, average rating given | veg / vegan / non-veg, rice-based |
| average rating per cuisine | price bucket, prep-time bucket |
| average order value bucket | average rating, number of ratings |
| stated spice tolerance | ingredient tags (multi-hot) |

Neither tower ever sees an id. That is what lets the model score a customer who
has ordered nothing and a dish nobody has rated — every input is knowable at
signup or at the moment a dish goes on the menu.

The final layer of each tower is **linear, not ReLU**. A ReLU there confines
every embedding to the non-negative orthant, where no two vectors can have a
cosine below zero, and the model loses its only way to express "this person
actively dislikes this".

### The scale problem

L2-normalising both towers makes the output a cosine similarity, bounded to
[−1, 1]. Training that against 1–5 star ratings with MSE is not merely
inaccurate — it is *unsatisfiable*, and the optimiser distorts the embedding
space chasing a target the output cannot reach.

So targets are mapped onto the output's own range and mapped back afterwards:

```
train on   y = (rating − 3) / 2      ∈ [−1, 1]
predict    rating = 2·(v_u · v_m) + 3 ∈ [1, 5]
```

This leaves the content score in **star units**, which is what makes the hybrid
blend below an addition of two quantities that mean the same thing.

The implicit path has the mirror problem: `sigmoid` of a value bounded to
[−1, 1] can only produce probabilities in [0.27, 0.73], so the model could never
express confidence. A single learnable scale on the logit — the temperature term
from contrastive learning — fixes it.

### Serving

`v_m` is precomputed for the whole catalogue at training time and cached; only
`v_u` is computed per request. The user tower is exported as numpy weights and
the forward pass is three matrix multiplies, so **the API serves this model
without TensorFlow installed**. Verified against TensorFlow on 50 users: maximum
difference 1.8e-07.

The item tower is exported too. Without it, a dish added after training would
have no embedding at all — defeating the exact case the model exists for — so
anything missing from the cache is embedded on demand.

---

## Phase C — Hybrid scoring

```
score = α · collaborative + (1 − α) · content
```

Both terms are in star units, by construction. Blending a 1–5 rating with a
raw −1…1 cosine adds quantities on different scales and lets whichever happens
to have the larger magnitude dominate, for reasons unconnected to quality.

**α is computed per candidate, not read from config once.** It is forced to 0 —
content only — in four situations:

| condition | why |
| --- | --- |
| user has < 5 ratings | their learned parameters were shrunk to nothing; a collaborative score for them carries no information |
| item has < 3 ratings | same argument on the item side, and it applies *per candidate* — one cold dish among warm ones is scored differently from its neighbours |
| the model has never seen this user | true of everyone who signed up since the last training run |
| the model has never seen this item | true of every dish added since |

The last two are the ones a static configuration value cannot express, and they
are why the decision lives in `effective_alpha()` rather than in a settings
file. Symmetrically, with no content model available α is forced to 1: falling
back to a collaborative-only score beats discarding a usable one.

α itself is selected on a validation split, not chosen by taste. On this
dataset it comes out at **0.2**, which leans heavily on the content model —
see [`EVALUATION.md`](EVALUATION.md) for the measurement and for why that
ordering is probably an artifact of synthetic data.

---

## Retrieval and ranking

Scoring every dish for every request works at 300 items and stops working long
before a real chain's catalogue. So the expensive model-backed stage only ever
sees a shortlist built by cheap indexed queries.

```
RETRIEVAL  (~150 candidates, indexed queries)
├─ the 10 nearest neighbours of each of the user's last 10 ordered dishes
├─ the top 10 dishes in each of their 3 most-ordered cuisines
└─ the 20 most popular dishes overall          ← the floor
   merge, de-duplicate, drop anything unavailable or already rated

RANKING  (hybrid model)
   score every candidate, sort descending, then:
   ├─ drop anything ordered in the last 24 hours
   ├─ add +0.25 stars to promoted dishes
   └─ return the top N
```

The three sources are in descending order of how personal they are, and a dish
surfaced by more than one keeps the first — so the explanation shown to the
customer is the strongest true thing that can be said about it.

**The popularity floor is what guarantees a brand-new customer never receives an
empty feed.** It is the one source that requires no history at all.

The promotion boost is deliberately small. +0.25 stars will lift a dish past one
it was narrowly behind; it will not float a bad dish over one that is a full
star better. There is a test pinning both halves of that.

Measured on the seeded data: retrieval 9.2 ms, ranking 6.7 ms, 25.6 ms end to
end uncached, 0.18 ms served from cache. A warm user retrieves 119 candidates;
a brand-new one retrieves 20, all from the floor, and every one of them is
scored by the content model.
