# Link Prediction on Node Pairs

## **Table of Contents**
- [Introduction](#introduction)
- [Environment](#environment)
- [Implementation](#implementation)
  - [Feature Extraction](#feature-extraction)
- [Code](#code)
- [How to run my code?](#how-to-run-my-code)
- [Another idea: Using Neural Network Model to make binary classification](#another-idea-using-neural-network-model-to-make-binary-classification)
  - [Features I used to train the model](#features-i-used-to-train-the-model)
  - [A Simple Model I built](#a-simple-model-i-built)
  - [Code of this method](#code-of-this-method)
  - [How to run my code?](#how-to-run-my-code?)


# Introduction


In this project, I tried many methods to predict the links between node pairs, such as logistic regression, neural networks, and random forest. However, these methods did not work well; the score on the public testing set fell between 0.5 and 0.6, which sucks, there might be some misunderstood

After much trial and error, eventually, I found a vital algorithm that can get a higher score of about 0.7: Weakly Connected Components (WCC). With the help of this algorithm, I don't even need to build a model to predict the link between nodes.

# Environment

- macOS Ventura 13.3
- Python 3.10
- Conda 23.1.0

<div style="page-break-after: always;"></div>

# Implementation

## Feature Extraction

The provided training data set comprises `node1`, `node2`, and `label`. To create the graph of it, I will have to separate node pairs depending on `label` and choose those pairs with `label` 1. With the graph, we can glimpse the whole dataset with other metrics calculated by the graph info.

### Weakly Connected Components (WCC)

According to [neo4j](https://neo4j.com/docs/graph-data-science/current/algorithms/wcc/), the Weakly Connected Components (WCC) algorithm finds sets of connected nodes in directed and undirected graphs. Two nodes are connected if a path exists between them—the set of all connected nodes forms a component. In contrast to Strongly Connected Components (SCC), the direction of relationships on the path is ignored in WCC.

The idea of using it is just a guess. First, I will have to check the input pairs are in the graph. Then check if they have an edge; if yes, let’s say the node pairs are in the same WCC, and return `1`; if not, we can compute the shortest path length between them. If we get the shortest path length, we can **ambiguously** predict they are in the same WCC, and return `1`; if not, return `-1`.

Determining whether node pairs are in the same weakly connected component (WCC) is **not always accurate**. While a path between two nodes indicates they are in the same WCC, the absence of a path does not necessarily mean they are not in the same WCC. Also, it is not directly related to feature extraction.

<div style="page-break-after: always;"></div>

## Code
Please refer to the [notebook](https://github.com/ghnmqdtg/link-prediction-on-node-pairs/blob/main/link_prediction_wcc.ipynb) for more information.


## How to run my code?

It’s recommended that you use a miniconda environment to run my code.

1. Create a conda env.
    
    ```bash
    $ conda create --name <env_name> python=3.10
    ```
    
2. Activate the env you’ve just created.
    
    ```bash
    $ conda activate <env_name>
    ```
    
3. Install all the required packages.
    
    ```bash
    $ pip install -r requirement.txt
    ```
    
4. Modify the config to load your custom CSV file path
5. Run the python file with the relative path to the test dataset. (Default is the public dataset)
    
    ```bash
    $ python link_prediction_wcc.py
    ```
    
<div style="page-break-after: always;"></div>

# Another idea: Using Neural Network Model to make binary classification

This part shows how I used many different features to train the neural network model to make link predictions. Although it doesn't reach the baseline, sharing my idea is worthwhile; maybe, I will figure it out someday.

## Features I used to train the model
1. **Number of followers, followees, inter followers and inter followees**
2. **Number of follows back**
3. **Degree Centrality**
    
    The degree centrality of a node is simply its degree—the number of edges it has. The higher the degree, the more central the node is.
    
4. **Betweenness Centrality**
    
    Betweenness centrality is a way of detecting the amount of influence a node has over the flow of information in a graph.
    
5. **Clustering Coefficient**
    
    A clustering coefficient is **a measure of the degree to which nodes in a graph tend to cluster together**.
    
6. **Jaccard distance of followers and followees**
    
    $$J(A, B) = \frac{|A \cap B |}{|A \cup B |}$$

    The [Jaccard Index](https://en.wikipedia.org/wiki/Jaccard_index) measures the similarity between finite sets and is defined as the intersection size divided by the union of the sample sets. It measures the similarity between sample sets complementary to the Jaccard index and is obtained by subtracting the Jaccard index by 1.
    
    $$d_j(A, B) = 1-J(A, B)$$

7. **Cosine distance of followers and followees**
    
    Here I use is one of the definition of cosine similarity named [Otsuka–Ochiai coefficient](https://en.wikipedia.org/wiki/Cosine_similarity#:~:text=Otsuka%E2%80%93Ochiai%20coefficient%5Bedit%5D):
    
    $$K = \frac{|A \cap B |}{\sqrt{|A| \times |B|}}$$

    Here, $A$ and $B$ are sets, and $|A|$ is the number of elements in $A$. If sets are represented as bit vectors, the Otsuka–Ochiai coefficient can be seen to be the same as the cosine similarity.
    
8. **Adar index**
    
    The Adamic-Adar index is a measure to predict links in a social network, according to the amount of shared links between two nodes. [It is defined as the sum of the inverse logarithmic degree centrality of the neighbours shared by the two nodes where is the set of nodes adjacent to.](https://en.wikipedia.org/wiki/Adamic%E2%80%93Adar_index)
    
    $$A(x, y)=\sum_{u \in N(x) \cap N(y)} \frac{1}{log|N(u)|}$$

9. **Belongs to same WCC**
    
    [As I mentioned earlier.](https://www.notion.so/58141a9b6d764779a79526266c874c5b)
    
10. **Shortest path**
    
    Shortest path is the path between two nodes such that the sum of their weights is minimum.
    
11. **Page Rank**
    
    Page Rank is an algorithm used by Google to rank webpages in their search engine results. It measures the importance of a webpage based on the number and quality of links to that page. **The more links a page has, the more important it is considered to be**.
    
12. **Katz Centrality**
    
    Katz centrality is a measure used to [measure the relative degree of influence of an node within a social network](https://en.wikipedia.org/wiki/Katz_centrality). Unlike typical centrality measures which consider only the shortest path between two nodes, Katz centrality measures influence based on the entire number of walks between two nodes.
    
13. **HITS**
    
    [Hyperlink-Induced Topic Search (HITS)](https://www.geeksforgeeks.org/hyperlink-induced-topic-search-hits-algorithm-using-networkx-module-python/) is a link analysis algorithm that rates nodes based on two scores, a hub score and an authority score. The authority score estimates the importance of the node within the network. The hub score estimates the value of its relationships to other nodes.

<div style="page-break-after: always;"></div>

## A Simple Model I built
```
model: BinaryClassification(
  (layer_1): Linear(in_features=28, out_features=64, bias=True)
  (layer_2): Linear(in_features=64, out_features=64, bias=True)
  (layer_out): Linear(in_features=64, out_features=1, bias=True)
  (relu): ReLU()
  (dropout): Dropout(p=0.1, inplace=False)
  (batchnorm1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True,
track_running_stats=True)
  (batchnorm2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True,
track_running_stats=True)
)
```


## Code of this method
Please refer to the [notebook](https://github.com/ghnmqdtg/link-prediction-on-node-pairs/blob/main/link_prediction_neural_network.ipynb) for more information.

## How to run my code?

It’s recommended that you use a miniconda environment to run my code.

1. Create a conda env.
    
    ```bash
    $ conda create --name <env_name> python=3.10
    ```
    
2. Activate the env you’ve just created.
    
    ```bash
    $ conda activate <env_name>
    ```
    
3. Install all the required packages.
    
    ```bash
    $ pip install -r requirement.txt
    ```
    
4. Modify the config to load your custom CSV file path
5. Run the python file with the relative path to the test dataset. (Default is the public dataset)
    
    ```bash
    $ python link_prediction_neural_network.py
    ```