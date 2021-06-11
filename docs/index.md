# Introduction

Propius is latin for *closer*. Propius is a simple tool to uncover similar items in a dataset. In terms of distance, similar items tend to be closer, that's why _propius_ in latin.

Its main feature is to allow for extracting similar items over a big data volume by using correlation between items over sparse data structures which use less space and memory.

It has two main components. First, the similarity model which is in charge of finding correlations between items based on how they happen together across data. Second, once model training is completed, Propius is able to store similarities so they can be retrieved later via REST API allowing to integrate this to different systems (e.g. recommender systems) transparently.


# How does it work?

