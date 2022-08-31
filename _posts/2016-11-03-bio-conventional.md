---
layout: distill
title: The Search for Biologically Plausible Neural Computation — Part 1
description: the conventional approach
date: 2016-11-03
categories: biologically-plausible neural-computation

authors:
  - name: Dmitri 'Mitya' Chklovskii
    url: "https://scholar.google.com/citations?user=7Bgb5TUAAAAJ&hl=en"
    affiliations:
      name: Flatiron Institue

toc:
  - name: A Sketch of a Biological Neuron
  - name: Primary Biological Constraints
  - name: Single-neuron Online Principal Component Analysis (PCA)
  - name: A Normative Theory
  - name: Deriving a Single-neuron Online PCA using the Reconstruction Approach
  - name: The Reconstruction Approach Fails for Multi-neuron Networks

---
Inventors of the original artificial neural networks (NNs) derived their inspiration from biology. However, as artificial NNs progressed, their design was less guided by neuroscience facts. Meanwhile, progress in neuroscience has altered our conceptual understanding of neurons. Consequently, we believe that many successful artificial NNs resemble natural NNs only superficially violating fundamental constraints imposed by biological hardware.

The wide gap between the artificial and natural NN designs raises intriguing questions: What algorithms underlie natural NNs? Can insights from biology help build better artificial NNs?

This is the first of a series of posts aimed at explaining recent progress made by my collaborators and myself towards biologically plausible NNs. Such networks can serve both as models of natural NNs and as general purpose artificial NNs. We have found that respecting biological constraints actually helps development of artificial NNs by guiding design decisions.

In this post, I cover the background material, going back several decades. I sketch a biological neuron, introduce primary biological constraints, and discuss the conventional approach to deriving artificial NNs. I will show that while the conventional approach generates a reasonable algorithmic model of a single biological neuron, multi-neuron networks violate biological constraints. In future posts we will see how to fix that.

## A Sketch of a Biological Neuron

Here is the minimum biological background needed to understand the rest of the post.

A biological neuron receives signals from multiple neurons, computes their weighted sum and generates a signal transmitted to multiple neurons, Figure 1. Each neuron’s signaling activity is quantified by the firing rate, which is a nonnegative real number that varies over time. Each synapse scales the input from the corresponding upstream neuron onto the receiving neuron by its weight. The receiving neuron sums scaled inputs, i.e. computes the inner product of the upstream activity vector and the synaptic weight vector. The inner product passes through a nonlinearity called the activation function and the output is transmitted to downstream neurons.

Synaptic weight changes over time, typically, on a slower time scale than neuronal signals. The weight depends on neuronal signals per so-called learning rules. For example, in commonly used [Hebbian learning rules](https://en.wikipedia.org/wiki/Hebbian_theory), synaptic weight is proportional to the correlation between the activities of the two neurons a synapse connects, i.e. pre- and postsynaptic.

## Primary Biological Constraints

To determine which algorithmic models in this post are biologically plausible, we can focus on a few key biological constraints.

Biologically plausible algorithms must be formulated in the online (or streaming), rather than offline (or batch), setting. This means that input data are streamed to the algorithm sequentially, one sample at a time, and the corresponding output must be computed before the next input sample arrives. The output communicated to downstream neurons cannot be modified in the future. A neuron cannot store individual past inputs or outputs except in a highly compressed format limited to synaptic weights and a few state variables.

In biologically plausible NNs, learning rules must be local. This means that the synaptic weight update may depend on the activities of only the two neurons a synapse connects, as for example, in Hebbian learning. Activities of other neurons are not physically available to a synapse and therefore including them into learning rules would be biologically implausible. Modern artificial NNs, such as backpropagation-based deep learning networks, rely on nonlocal learning rules.

Our initial focus is on unsupervised learning. This is not a hard constraint, but rather a matter of priority. Whereas humans are clearly capable of supervised learning, most of our learning tasks lack big labeled datasets. On the mechanistic level, most neurons lack a clear supervision signal.


## Single-neuron Online Principal Component Analysis (PCA)

In 1982, [Oja proposed](https://www.semanticscholar.org/paper/Simplified-neuron-model-as-a-principal-component-Oja/3e00dd12caea7c4dab1633a35d1da3cb2e76b420?p2df) modeling a neuron by an online PCA algorithm. PCA is a workhorse of data analysis used for dimensionality reduction, denoising, and latent factor discovery. Therefore, Oja’s seminal paper established that biological processes in a neuron can be viewed as the steps of an online algorithm solving a useful computational objective.

Oja’s single-neuron online PCA algorithm works as follows. At each time step, $$t$$, it receives an input data sample, $${\bf x}_t$$, computes and outputs the corresponding top principal component value, $${\bf y}_t$$:

$$y_t \leftarrow {\bf w} _{t-1}^\top {\bf x}_t. \qquad \qquad \qquad (1.1)$$

Here and below lowercase boldfaced letters designate vectors. Then the algorithm updates the (normalized) feature vector,

$${\bf w} _t \leftarrow {\bf w} _{t-1}+ \eta ( {\bf x} _t- {\bf w} _{t-1} y_t  ) y_t. \qquad \qquad (1.2)$$

The feature vector, w, converges to the eigenvector of input covariance if data are drawn i.i.d from a stationary distribution.

The steps of the Oja algorithm (1.1-1.2) correspond to the operations of the biological neuron. If the input vector is represented by the activities of the upstream neurons, (1.1) represents weighted summation of the inputs by the output neuron. If the activation function is linear the output, $${\bf y}_t$$, is simply the weighted sum. The update (1.2) is a local Hebbian synaptic learning rule. The first term of the update is proportional to the correlation of the pre- and postsynaptic neurons’ activities and the second term, also local, normalizes the synaptic weight vector.

## A Normative Theory

Next, we would like to build on Oja’s insightful identification of biological processes with the steps of the online PCA algorithm by computing multiple principal components using multi-neuron NNs and including the activation nonlinearity.

Instead of trying to extend the Oja model heuristically, we take a more systematic, so-called normative approach. In this approach, a biological model is viewed as the solution of an optimization problem. Specifically, we postulate an objective function motivated by a computational principle, derive an online algorithm optimizing such objective, and map the steps of the algorithm onto biological processes.

Having such normative theory allows us to navigate through the space of possible algorithmic models in a more efficient and systematic way. Mathematical compactness of objective functions facilitates generating new models and weeding out inconsistent ones. This is similar to the Hamiltonian approach in physics which leverages natural symmetries and safeguards against the violation of the first law of thermodynamics (energy conservation).

## Deriving a Single-neuron Online PCA using the Reconstruction Approach
To build a normative theory, we first need to derive Oja’s single-neuron online algorithm by solving an optimization problem. What objective function should we choose for online PCA? Historically, neural activity has been often viewed as representing each data sample, $${\bf x}_t$$, by the feature vector, $$w$$, scaled by the output, $${\bf y}_t$$, Figure 2. Such reconstruction approach is naturally formalized as the minimization of the reconstruction (or coding) error:

$$\min_{\| {\bf w} \|=1} {\sum \limits_{t=1}^{T} \min_{ y_t} {\| {\bf x}_t-{\bf w} y_t \| ^2_2}}. \qquad \qquad \qquad \quad (1.3)$$

In the offline setting, optimization problem (1.3) is solved by PCA: the optimum w is the eigenvector of input covariance corresponding to the top eigenvalue and the optimum output, y, is the first principal component.

In the online setting, (1.3) can be solved by alternating minimization, which has been a subject of recent analysis. After the arrival of each data point, $${\bf x}_t$$ , the algorithm computes optimum output, $${\bf y}_t$$, while keeping the feature vector, $$w_{t−1}$$, computed at the previous time step, fixed. By using calculus, one finds that the optimum output is given by (1.1). Then, the algorithm minimizes the total reconstruction error with respect to the feature vector while keeping all the outputs fixed. Again, resorting to calculus, one finds (1.2).

Thus, the single-neuron online PCA algorithm may be derived using the reconstruction approach. To compute multiple principal components, we need to extend this success to multi-neuron networks.

## The Reconstruction Approach Fails for Multi-neuron Networks

Though the reconstruction approach yields a multi-component online PCA algorithm, the corresponding NNs are not biologically plausible.

Extension of the reconstruction error objective from single to multiple output components is straightforward - each scalar, $${\bf y}_t$$, is replaced by a vector, $${\bf y}_t$$:

$$\min_{\rm diag({\bf W}^\top {\bf W})={\bf I}} {\sum \limits_{t=1}^{T} \min_{ {\bf y}_t} {\| {\bf x}_t-{\bf W} {\bf y}_t \| ^2_2}}. \qquad \qquad \qquad \quad (1.4)$$

Here matrix W comprises column-vectors corresponding to different features. As in the single- neuron case this objective can be optimized online by alternating minimization. After the arrival of data sample, $${\bf x}_t$$, the feature vectors are kept fixed while the objective (1.4) is minimized with respect to the principal components by iterating the following update until convergence:

$${\bf y}_t \leftarrow {\bf W} _{t-1}^\top {\bf x}_t - {\bf W} _{t-1}^\top {\bf W} _{t-1} {\bf y}_t . \qquad \qquad \qquad (1.5)$$

Minimizing the total objective with respect to the feature vectors for fixed principal components yields the following update:

$${\bf W} _t \leftarrow {\bf W} _{t-1}+ \eta ( {\bf x} _t- {\bf W} _{t-1} {\bf y}_t ) {\bf y}_t^\top  \cdot \qquad \qquad (1.6)$$

As before, in NN implementations of algorithm (1.5-1.6), feature vectors are represented by synaptic weights and principal components by the activities of output neurons. Then (1.5) can be implemented by a single-layer NN, Figure 3, in which activity dynamics converges faster than the time interval between the arrival of successive data samples.

However, implementing update (1.6) in the single-layer NN architecture, Figure 3, requires nonlocal learning rules making it biologically implausible. Indeed, the last term in (1.6) implies that updating the weight of a synapse requires the knowledge of output activities of all other neurons which are not available to the synapse. Moreover, the matrix of lateral connection weights, $$−{\bf W}^\top_{t−1}W_{t−1}$$, in the last term of (1.5) is computed as a Grammian of feedforward weights, clearly a nonlocal operation. This problem is not limited to PCA and arises in networks of nonlinear neurons as well.

Rather than deriving learning rules from a principled objective, many authors constructed biologically plausible single-layer networks using local learning rules, Hebbian for feedforward and anti-Hebbian (meaning there is a minus sign in front of the correlation-based synaptic weight as for the last term in (1.5)). However, in my view, abandoning the normative approach creates more problems than it solves.

I have outlined how the conventional reconstruction approach fails to generate biologically plausible multi-neuron networks for online PCA. In the next post, I will introduce an alternative approach that overcomes this limitation. Moreover, this approach suggests a novel view of neural computation leading to many interesting extensions.

_Acknowledgement: I am grateful to Sanjeev Arora for his support and encouragement as well as to Cengiz Pehlevan, Leo Shklovskii, Emily Singer, and Thomas Lin for their comments on the earlier versions._