# Introduction

Propius is latin for *closer*. Propius is a simple tool to uncover similar items in a dataset. In terms of distance, similar items tend to be closer, that's why _propius_ in latin.

Its main feature is to allow for extracting similar items over a big data volume by using correlation between items over sparse data structures which use less time and memory space.

It has two main components. First, the similarity model which is in charge of finding correlations between items based on how they happen together across data. Second, once model training is completed, Propius is able to store similarities so they can be retrieved later via REST API allowing to integrate this to different systems (e.g. recommender systems) transparently.

<br />
# How does it work?

Propius take advantage of [SciPy Sparse Module](https://docs.scipy.org/doc/scipy/reference/sparse.html) to build a correlation coefficients matrix to model similarities between items as distances between vectors in a _i_-dimensional space where _i_ represent an item.

Using this correlation coefficients matrix Propius is able to calculate kNN per each unique item and store each similarity score in a local db (sqlite) to retrieve similar items without the need to keep the correlation matrix in memory and returning result faster at the same time.


