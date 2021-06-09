# Propius

Propius is latin for *closer*. Propius is a simple tool to uncover similar items in a dataset. In terms of distance, similar items tend to be closer, that's why _propius_ in latin.

Its main feature is to allow for extracting similar items over a big data volume by using correlation between items over sparse data structures which use less space and memory.

# System Design

## Requirements


1. The system should find similar items to a particular item in a dataset
2. The system should provide a dataset format to use for training similarity algorithm

## Performance Estimation
...

## System Interface

1. API
	- Allow to query similar items based on a specific item
	- Allow to query all resources available
	- Allow to train similarity model for a resource

2. Web Front End
	- Using system backend api, system show list all resources
	- Using system backend api, system show create and delete resources

## Components

...
