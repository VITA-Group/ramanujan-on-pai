## Pruning methods:

**RP:** random pruning

**OMP:** oneshot pruning, magnitude pruning

**GMP**: To prune, or not to prune: exploring the efficacy of pruning for model compression

**TP:** Detecting Dead Weights and Units in Neural Networks, Page19 Table2.1 Taylor1Scorer (adding abs in our implementation) 

**SNIP:** SNIP: Single-shot network pruning based on connection sensitivity

**GraSP:** Picking winning tickets before training by preserving gradient flow

**SynFlow:** Pruning neural networks without any data by iteratively conserving synaptic flow



### Details:

setting for SNIP, GraSP and SynFlow can follow paper: Pruning Neural Networks at Initialization: Why are We Missing the Mark?

To be more specific: SNIP, GraSP using oneshot pruning, with the number of images equals 10*classes

SynFlow using iterative pruning of 100 iterations, using one image with all values equal 1





## Code Details: 

pruning methods implemented in **pruning_utils.py**

before pruning, please modify the layers in the original model with Linear and Conv defined in **layers.py**

**example.py** provides an simple examples

