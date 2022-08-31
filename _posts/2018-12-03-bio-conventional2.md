---
layout: distill
title: The Search for Biologically Plausible Neural Computation — Part 2
description: a similarity-based approach
date: 2018-12-03
categories: biologically-plausible neural-computation

authors:
  - name: Dmitri 'Mitya' Chklovskii
    url: "https://scholar.google.com/citations?user=7Bgb5TUAAAAJ&hl=en"
    affiliations:
      name: Flatiron Institue
  - name: Cengiz Pehlevan
    url: "https://scholar.google.com/citations?user=veDLTPEAAAAJ"
    affiliations:
      name: Harvard University
toc:
  - name: Similarity-matching objective function
  - name: Variable substitution trick
  - name: Online algorithm and neural network
  - name: Other similarity-based objectives and linear networks
  - name: Nonnegative similarity-matching objective and a nonlinear network
  - name: Similarity-based algorithms as general-purpose tools

---
This is the second post in a series reviewing recent progress in designing artificial neural networks (NNs) that resemble natural NNs not just superficially, but on a deeper, algorithmic level. In addition to serving as models of natural NNs, such networks can serve as general-purpose machine learning algorithms. Respecting biological constraints, viewed naively as a handicap in developing competitive general-purpose machine learning algorithms, can instead facilitate the development of artificial NNs by restricting the search space of possible algorithms.

In the previous post, we focused on the constraints that must be met for an unsupervised algorithm to be biologically plausible. For an algorithm to be implementable as a NN, it must be formulated in the online setting. In the corresponding NN, synaptic weight updates must be local, i.e. depend on the activity of only two neurons that the synapse connects. Then, we demonstrated that deriving NNs for dimensionality reduction in a conventional way - by minimizing the reconstruction error - results in multi-neuron networks with biologically implausible non-local learning rules.

In this post, we propose a different objective function which we term similarity matching. From this objective function, we derive an online algorithm implementable by a NN with local learning rules. Then, we introduce other similarity-based algorithms which include more biological features such as different neuron classes and nonlinear activation functions. Finally, we review similarity-matching algorithms with state-of-the-art performance.

## Similarity-matching objective function

We start by stating an objective function that will be used to derive NNs for linear dimensionality reduction. Let ${\bf x_t} \in R^n, t=1,…T$, be a set of data points (inputs) and ${\bf y_t} \in R^k, t=1,…T, (k<n)$ be their learned representation (outputs). The similarity of a pair of inputs, ${\bf x_t}$ and ${\bf x_t \'}$, can be defined as their dot-product, ${\bf x_t}$⊤${\bf x_t\'}$. Analogously, the similarity of a pair of outputs is {\bf y_t}⊤{\bf y_t}′. Similarity matching, as its name suggests, learns a representation where the similarity between each pair of outputs matches that of the corresponding inputs:

$$
\min_{ {\bf y}_1,\ldots,{\bf y}_T}  \frac{1}{T^2}  \sum_{t=1}^T  \sum_{t'=1}^T \left({\bf x}_t^\top {\bf x}_{t'} - {\bf y}_t^\top {\bf y}_{t'} \right)^2. \qquad\qquad (2.1)
$$

This offline objective function, previously employed for multidimensional scaling, is optimized by the projections of inputs onto the principal subspace of their covariance, i.e. performing PCA up to an orthogonal rotation. Moreover, (2.1) has no local minima other than the principal subspace solution.

The similarity-matching objective (2.1) may seem like a strange choice for deriving an online algorithm implementable by a NN. In the online setting, inputs are streamed to the algorithm sequentially and each output must be computed before the ne${\bf x_t}$ input arrives. Yet, in (2.1), pairs of inputs and outputs from different time points interact with each other. In addition, whereas ${\bf x_t}$ and ${\bf y_t}$ could be interpreted as inputs and outputs to a network, unlike in the reconstruction approach (1.4), synaptic weights do not appear explicitly in (2.1).


## Variable substitution trick
Both of the above concerns can be resolved by a simple math trick akin to completing the square. We first focus on the cross-term in (2.1), which we call similarity alignment. By re-ordering the variables and introducing a new variable, ${\bf W} \in R^{k \times n}$, we obtain:

$$
- \frac{1}{T^2}\sum_{t=1}^T \sum_{t'=1}^T  {\bf y}_{t}^\top {\bf y}_{t'} {\bf x}_t^\top {\bf x}_{t'}= - \frac{1}{T^2}\sum_{t=1}^T  {\bf y}_{t}^\top  \left[ \sum_{t'=1}^T {\bf y}_{t'} {\bf x}_{t'}^\top \right ] {\bf x}_t
$$

$$
\qquad \qquad \qquad \qquad\qquad  =   \min_{ {\bf W} \in \mathbb{R}^{k\times n}}  -\frac{2}{T} \sum_{t=1}^T{\bf y}_t^\top {\bf W} {\bf x}_t + {\rm Tr } {\bf W}^\top {\bf W}.\; (2.2)
$$

To prove the second identity, find optimal ${\bf W}$ by taking a derivative of the expression on the right with respect to ${\bf W}$ and setting it to zero, and then substitute the optimal ${\bf W}$ back into the expression. Similarly, for the quartic ${\bf y_t}$ term in (2.1):

$$
\frac{1}{T^2}\sum_{t=1}^T \sum_{t'=1}^T  {\bf y}_{t}^\top {\bf y}_{t'} {\bf y}_t^\top {\bf y}_{t'}= \frac{1}{T^2}\sum_{t=1}^T  {\bf y}_{t}^\top  \left[ \sum_{t'=1}^T {\bf y}_{t'} {\bf y}_{t'}^\top \right ] {\bf y}_t
$$

$$
\qquad \qquad \qquad \qquad\qquad  =   \max_{ {\bf M} \in \mathbb{R}^{k\times k}}  \frac{2}{T} \sum_{t=1}^T{\bf y}_t^\top {\bf M} {\bf y}_t - {\rm Tr }  {\bf M}^\top {\bf M}.\quad(2.3)
$$

By substituting (2.2) and (2.3) into (2.1) we get:

$$
\min_{ {\bf W}\in \mathbb{R}^{k\times n}}\max_{ {\bf M}\in \mathbb{R}^{k\times k}}  \frac{1}{T} \sum_{t=1}^T \left[2 {\rm Tr}\left({\bf W}^\top{\bf W}\right) - {\rm Tr}\left({\bf M}^\top{\bf M}\right) +  \min_{ {\bf y}_t\in \mathbb{R}^{k\times 1}}  l_t({\bf W},{\bf M},{\bf y}_t)\right],\; (2.4)
$$

where

$$
l_t({\bf W},{\bf M},{\bf y}_t)=-4{\bf x}_t^\top{\bf W}{\bf y}_t + 2{\bf y}_t^\top{\bf M}{\bf y}_t.\qquad\qquad (2.5)
$$

In the resulting objective function, (2.4),(2.5), optimal outputs at different time steps can be computed independently, making the problem amenable to an online algorithm. The price paid for this simplification is the appearance of the minimax optimization problem in variables, ${\bf W}$ and ${\bf M}$. Minimization over  ${\bf W}$ aligns output channels with the greatest variance directions of the input and maximization over ${\bf M}$ diversifies the output channels. The competition between the two in a gradient descent/ascent algorithm results in the principal subspace projection which is the only stable fixed point of the corresponding dynamics.

## Online algorithm and neural network
Now, we are ready to derive an algorithm for optimizing (2.1) online. First, by minimizing (2.5) with respect to ${\bf y_t}$ while keeping ${\bf W}$ and $ {\bf M} $ fixed we get the dynamics for the output variables:

$$
\dot{\bf y}_t={\bf W} {\bf x}_t-{\bf M} {\bf y}_t.\qquad\qquad (2.6)
$$

To find ${\bf y_t}$ after the presentation of the corresponding input, ${\bf x_t}$, (2.6) is iterated until convergence.

After the convergence of ${\bf y_t}$ we update ${\bf W}$ and $ {\bf M} $ by gradient descent of (2.2) and gradient ascent of (2.3) respectively:

$$
W_{ij} \leftarrow W_{ij} + \eta \left(y_i{\bf x_j}-{\bf W}_{ij}\right), \qquad   M_{ij} \leftarrow M_{ij} + \eta \left(y_iy_j-M_{ij}\right). \qquad (2.7)
$$

Algorithm (2.6),(2.7), first derived [here](https://www.researchgate.net/publication/273003026_A_HebbianAnti-Hebbian_Neural_Network_for_Linear_Subspace_Learning_A_Derivation_from_Multidimensional_Scaling_of_Streaming_Data), can be naturally implemented by a biologically plausible NN, Figure 1. Here, activity (firing rate) of the upstream neurons corresponds to input variables. Output variables are computed by the dynamics of activity (2.6) in a single layer of neurons. Variables ${\bf W}$ and $ {\bf M} $ are represented by the weights of synapses in feedforward and lateral connections respectively. The learning rules (2.7) are local, i.e. the weight update, ΔWij, for the synapse between jth input neuron and ith output neuron depends only on the activities, ${\bf x_j}$, of jth input neuron and, ${\bf y_i}$, of ith output neuron, and the synaptic weight. In neuroscience, learning rules (2.7) for ${\bf W}$ and ${\bf M}$ are called Hebbian and anti-Hebbian respectively.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/bio2_Figure1.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 1: A Hebbian/Anti-Hebbian network derived from similarity matching.
</div>

To summarize, starting with the similarity-matching objective, we derived a Hebbian/anti-Hebbian NN for dimensionality reduction. The minimax objective can be viewed as a zero-sum game played by the weights of feedforward and lateral connections. This demonstrates that synapses with local updates can still collectively work together to optimize a global objective. A similar, although not identical, NN was proposed by Foldiak heuristically. The advantage of our normative approach is that the offline solution is known. Although no proof of convergence exists in the online setting, algorithm (2.6),(2.7) performs well in practice.

## Other similarity-based objectives and linear networks
We used the same framework to derive NNs for other computational tasks and incorporating more biological features. As the algorithm (2.6),(2.7) and the NN in Figure 1 were derived from the similarity-matching objective (2.1), they project data onto the principal subspace but do not necessarily recover principal components per se. To derive PCA algorithms we modified the objective function (2.1), [here](https://arxiv.org/abs/1511.09468) and (here)[https://arxiv.org/abs/1810.06966], to encourage orthogonality of ${\bf W}$. Such algorithms are implemented by NNs of the same architecture as in Figure 1 but with slightly different learning rules.

Although the similarity-matching NN in Figure 1 relies on biologically plausible local learning rules, it lacks biological realism in several other ways. For example, computing output requires recurrent activity that must settle faster than the time scale of the input variation, which is unlikely in biology. To respect this biological constraint, [we modified](https://arxiv.org/abs/1810.06966) the dimensionality reduction algorithm to avoid recurrency.

Another non-biological feature of the NN in Figure 1 is that the output neurons compete with each other by communicating via lateral connections. In biology, such interactions are not direct but mediated by interneurons. To reflect these observations, we modified the objective function by introducing a whitening constraint:

$$
\min_{ {\bf y}_1,\ldots,{\bf y}_T}  - \frac{1}{T^2}\sum_{t=1}^T \sum_{t'=1}^T  {\bf y}_{t}^\top {\bf y}_{t'} {\bf x}_t^\top {\bf x}_{t'}, \qquad {\rm s.t.} \quad \frac 1T \sum_{t=1}^T {\bf y}_t {\bf y}_t^\top = {\bf I}_k, \qquad (2.8)
$$

where Ik is the k-by-k identity matrix. Then, by representing the whitening constraint using Lagrange relaxation, we derived NNs where interneurons appear naturally - their activity is modeled by the Lagrange multipliers, z⊤tzt′ (Figure 2):

$$
\min_{ {\bf y}_1,\ldots,{\bf y}_T} \max_{ {\bf z}_1,\ldots,{\bf z}_T } - \frac{1}{T^2}\sum_{t=1}^T \sum_{t'=1}^T  {\bf y}_{t}^\top {\bf y}_{t'} {\bf x}_t^\top {\bf x}_{t'} + \frac{1}{T^2}\sum_{t=1}^T \sum_{t'=1}^T  {\bf z}_{t}^\top {\bf z}_{t'} \left({\bf y}_{t}^\top {\bf y}_{t'} -\delta_{t,t'}\right). \; (2.9)
$$

Notice how (2.9) contains the y-z similarity-alignment term similar to (2.2). We can now derive learning rules for the y-z connections using the variable substitution trick, leading to the network in Figure 2. For details of this and other NN derivations, see [here](http://papers.nips.cc/paper/5885-a-normative-theory-of-adaptive-dimensionality-reduction-in-neural-networks).

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/bio2_Figure2.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 2: A biologically-plausible NN for whitening inputs, derived from a constrained similarity-alignment cost function.
</div>

## Nonnegative similarity-matching objective and a nonlinear network

So far, we considered similarity-based NNs with linear neurons. However, biological neurons are not linear and many interesting computations require nonlinearity. A resolution to this discrepancy was suggested by the observation that the output of biological neurons is nonnegative (firing rate cannot be below zero). Hence, we modified the optimization problem by requiring that the output of the similarity-matching cost function (2.1) is nonnegative:

$$
\min_{ {\bf y}_1,\ldots,{\bf y}_T \geq 0}  \frac{1}{T^2}  \sum_{t=1}^T  \sum_{t'=1}^T \left({\bf x}_t^\top {\bf x}_{t'} - {\bf y}_t^\top {\bf y}_{t'} \right)^2. \qquad\qquad (2.10)
$$

Solutions of the optimization problem (2.10) are very different from PCA: They can cluster well-segregated data and extract sparse features from data. Understanding the nature of these solutions will be the topic of the next. post. For now, we note that (2.10) can be solved by the same online algorithm as (2.1) except that the output variables are projected onto the nonnegative domain. Such algorithm maps onto the same network as Figure 1 but with rectifying neurons (ReLUs), Figure 3A.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/bio2_Figure3.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 3: A) A nonlinear Hebbian/Anti-Hebbian network derived from nonnegative similarity matching. B) Stacked network for NICA. NSM - nonnegative similarity-matching.
</div>

Another problem solved by similarity-based networks is the nonnegative independent component analysis (NICA) which can be used for blind source separation. The problem is to recover independent and nonnegative sources from observing only their linear mixture. [Plumbley](https://www.researchgate.net/publication/3342750_Conditions_for_nonnegative_independent_component_analysis) showed that NICA can be solved in two steps, Figure 4. First, whiten the data to obtain an orthogonal rotation of the sources. Second, find an orthogonal rotation of the whitened sources that yields a nonnegative output, Figure 4. The first step can be implemented by the whitening network in Figure 2. The second step can be implemented by the nonnegative similarity-matching network, Figure 3A, because an orthogonal rotation does not affect dot-product similarities. Therefore, NICA is solved by stacking the whitening and the nonnegative similarity-matching networks, Figure 3B.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/bio2_Figure4.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 4: Illustration of the nonnegative independent component analysis algorithm. Two source channels (left) are linearly transformed to a two-dimensional mixture, which are the inputs to the algorithm (middle). Whitening (right) yields an orthogonal rotation of the sources. Sources are then recovered by solving the nonnegative similarity-matching problem. Green and red plus signs track two source vectors across mixing and whitening stag
</div>

## Similarity-based algorithms as general-purpose tools

While the derivation of the similarity-matching algorithm was motivated by constraints imposed by biology, the resulting algorithm performs well on large-scale data. A recent paper introduced an efficient modification of the similarity-matching algorithm and demonstrated its competitiveness with the state-of-the-art principal subspace projection algorithms in both processing speed and convergence rate. A package with implementations of these algorithms is [here](https://github.com/flatironinstitute/online_psp) and [here](https://github.com/flatironinstitute/online_psp_matlab).

In this blog post, we introduced linear and non-linear similarity-matching NNs that can serve as models of biological NNs and as general-purpose machine-learning tools. In the ne${\bf x_t}$ post, we will discuss the nature of the solutions to nonnegative similarity-based networks.
