# Clustering Synthetic Blobs: K-Means, Agglomerative, DBSCAN

[![CI](https://github.com/rohanreddy79/ML-Scatter-Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/rohanreddy79/ML-Scatter-Classification/actions/workflows/ci.yml)

**A short scikit-learn notebook that generates four Gaussian blobs with `make_blobs` and compares how K-means, single-link and complete-link hierarchical clustering, and DBSCAN recover them.**

Despite the repository name, this is an unsupervised *clustering* exercise, not a classification
model: there is no training label used by any of the algorithms. The notebook's Colab name is
`Lab_2_Final.ipynb`; the same lab with the discussion questions and written answers kept is in the
sibling repository `Random-Clusters`.

## Overview

`ML.ipynb` builds a 200-point, 2-feature dataset from four cluster centres
(`make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.6, random_state=50)`), plots it
coloured by the true blob, and then runs three families of clustering algorithms on the raw points
and scatters the result of each. It is a compact lab-style walkthrough using only scikit-learn,
NumPy and Matplotlib; each algorithm is called with the parameters the lab prescribes and the
output is a plot plus a printed cluster count, so the point is visual comparison rather than a
tuned model.

## What it does

- **Data**: `sklearn.datasets.make_blobs`, 200 samples, 4 centres, standard deviation 1.6, seeded
  with `random_state=50`; axes fixed to [-15, 15] in every plot so results line up.
- **Part 1, K-means**: `KMeans(n_clusters=3)` is fitted, `cluster_centers_` printed, and the three
  partitions scattered in red / black / blue. Two printed paragraphs discuss K = 3 and K = 4.
- **Part 2, hierarchical clustering**: `AgglomerativeClustering(n_clusters=4, linkage="single",
  metric="euclidean")` (MIN) and the same with `linkage="complete"` (MAX), each followed by a
  scatter coloured by label.
- **Part 3, DBSCAN**: `DBSCAN(eps=2, min_samples=28, metric="euclidean")`; noise points
  (label -1) are counted and left out of the plot.

## How it works

| File | Purpose |
| --- | --- |
| `ML.ipynb` | The whole lab: data generation, K-means, single/complete-link agglomerative, DBSCAN, plots |
| `requirements.txt` | Pinned scikit-learn 1.9.1, NumPy, Matplotlib and the nbformat/nbconvert/ipykernel runtime |
| `.github/workflows/ci.yml` | nbformat validation, headless execution, check of the printed cluster counts |

Pipeline: `make_blobs` -> NumPy array `points` -> one `fit_predict` per algorithm -> labels used to
colour a Matplotlib scatter. No scaling, train/test split or scoring metric is involved.

### Results reported in the notebook

- K-means with k = 3 learned centres at roughly (-5.51, -2.26), (-2.14, 7.58) and (0.05, -5.35),
  i.e. two of the four blobs are merged.
- Single-link agglomerative: "Found 4 total clusters!".
- Complete-link agglomerative: "Found 4 total clusters!".
- DBSCAN (`eps=2`, `min_samples=28`): "Found 4 total clusters!" and
  "49 points were dropped as noise."

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # scikit-learn 1.9.1, matplotlib, nbconvert, ipykernel
pip install notebook && jupyter notebook ML.ipynb   # interactive
```

Headless, without editing the committed notebook:

```bash
jupyter nbconvert --to notebook --execute ML.ipynb --output executed.ipynb
```

Verified on this machine (2026-10, Python 3.12.12, scikit-learn 1.9.1): the command above finished
in about 10 s (`[NbConvertApp] Writing ... bytes to executed.ipynb`) with no cell errors, and the
printed results were the same as the previously committed outputs, produced on scikit-learn < 1.4 (K-means
centres `[-5.50955539 -2.25816149]`, `[-2.13583883 7.57989854]`, `[0.05161133 -5.35489826]`,
"Found 4 total clusters!" for single-link, complete-link and DBSCAN, "49 points were dropped as
noise."). The committed notebook outputs are from that run. CI repeats the execution on every push.

## Changelog

- 2026-10: replaced the `affinity=` keyword (removed in scikit-learn 1.4) with `metric=` in the two
  `AgglomerativeClustering` calls and re-executed the notebook on scikit-learn 1.9.1 (identical
  cluster counts, centres and noise count); pinned requirements; added MIT license and CI.

## Limitations / Notes

- The K = 4 discussion paragraph is printed from a string, but the code only ever fits K-means with
  `n_clusters=3`; there is no K = 4 run or plot in the notebook.
- `KMeans` is not seeded, so cluster ordering (and, in principle, the centres) can vary between runs
  even though the data are fixed by `random_state=50`.
- No quantitative evaluation (silhouette, ARI against the true blob labels) is computed; the
  comparison is purely visual.
