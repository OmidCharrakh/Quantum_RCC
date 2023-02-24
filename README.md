# Quantum RCC
An extension of the RCC discovery algorithm to quantum scenarios

Causal discovery algorithms (CDAs) refer to computational techniques for inferring causal relationships between a set of variables from their statistical relationships. In recent years, researchers in the field of machine learning (ML) have developed powerful CDAs that can exploit ML tools to learn the direction of causality in various domains. Since these algorithms usually take probability distributions over classical variables as input, it is a nontrivial question how they are to be applied to probabilistic descriptions of quantum states such as density matrices. Here, we generalize a powerful classical CDA, called the [Randomized Causation Coefficient (RCC)](http://proceedings.mlr.press/v37/lopez-paz15.pdf), and apply it to several simulated quantum scenarios. In our scenarios, we consider the transmission of a quantum state between two qubits via various types of potentially noisy channels, and the aim is to learn the direction of the channel, and thus the causal direction, from the states of the qubits without explicitly performing interventions. We will show that the RCC confidently achieves this goal in simple networks (consisting of only two nodes) as well as more complex networks (having multi-nodes). 

The repository contains the codes for different stages of this project. This includes the codes for quantum simulations, several optimization steps, interpretational and visualization techniques. To overview the project's motivations, look at my [primary slides](https://github.com/OmidCharrakh/quantum_rcc/blob/main/Extra/project_slides.pdf).


# PipeLine (MultiNode Scenario)

1. Generate "org" channel data (8-Nodes graphs and the mechanism graph)

2. Fit a regression model on the mechanism data (nn_simulator)

3. Generate "sim" channel data via the nn_simulator (8-Nodes graphs)

4. Featurize the org & sim data 

5. Prepare the org & sim data
    - Get all triples of 8-Nodes graphs
    - Put data in separate dataframes
    - Store the causal directions in separate columns

6. Build a high-quality training dataset via **Active Learning**
    - Train a base classifier using initial and pool datasets
    - Take the dataset chosen by the base classifier 

7. Train and Tune an advanced classifier  

8. Prune the predicted graphs via
    - optimal percentile
    - penality penalty weight for number of predicted edges
    - hill-climbing algorithm, using nn_simulator 

### To be Done 
- Generate informative plots 
- Open ML's black box


### Further Works
- outcomes instead of rho => KME of (setting, outcome) using discrete RCC
- Work on entangled scenarios
- Estimate channel parameters MLE => Expectation-Minimization
- Investigate scenarios with different number of labs  
- GAN


### Archived
- Augmentation: additional scales, singe-scale 
- Two-stage clf: detect independent nodes and then oriente the edge between dependent nodes
- Colab 
- use 28 outputs with MSE reg
- Play around with z_is_triple, z_is_compressed 
- Create a NN for clf2 with one linear layer: taking the inverse_probs of clf1, returning the probs for each node 
- Instead of summing over triples, directly predict the class of x-y by setting z_is_compressed=True and train a clf on a mixture of 3 and 8 nodes
- balanced_sampling 1) theta is determined under condition wg_c1=wg_c2, 2) dirichlet and noisey, 3) using_percent_s is optimized, 4) each triple is annotated by with a 10-level label indicating the underlying triple type => stratify not only on y={0,1,2} but also on L={0,..,9} 
- sklearn multilabel clf

- **Simultaneous Learning:** 1) Train a classifier on 3-Nodes data, simultaneously validate it on 8-Nodes, 2) Use different loss criteria for training and validation

- **Ch-Parameter Estimation:** Estimate the physical parameters of the channel (estimation_torch)

- **Balanced Sampling:** 1) train a base classifier on combinations between two distributions: uniform and extremely imbalanced, 2) take the combination on which the base classifier has been more accurate 