Self-Pruning Neural Network on CIFAR-10

Problem Statement
Through this project i have implemented a self-pruning feed-forward neural network for CIFAR-10.  
The final goal was to train a model that learns which weights are unnecessary during training by associating each weight with a learnable gate parameter.

Model Design
I implemented a custom `PrunableLinear` layer instead of using `torch.nn.Linear` directly.  
Each layer contains:
- a standard weight matrix,
- a bias vector,
- a learnable `gate_scores` tensor with the same shape as the weight matrix.

During the forward pass, the gate scores are passed through a sigmoid function to obtain gate values between 0 and 1.  
The effective weight used by the layer is:

`pruned_weight = weight * sigmoid(gate_scores)`

This allows the network to gradually suppress unimportant connections during training.

Loss Function
The total training loss is:

`Total Loss = Classification Loss + λ × Sparsity Loss`

The classification loss is standard cross-entropy loss on CIFAR-10.  
The sparsity loss is based on the gate values across all prunable layers.

Why L1 on Sigmoid Gates Encourages Sparsity
An L1 penalty increases linearly with the value of each gate, so every active gate adds a direct cost to the loss. 
Because the gates are produced by a sigmoid, they remain between 0 and 1, and minimizing their L1-based penalty encourages many of them to become very small. [file:1]  
When a gate becomes close to 0, the corresponding weight contribution is effectively removed, which makes the network sparse. 

Experimental Setup
- Dataset: CIFAR-10 from `torchvision.datasets` 
- Model: Multi-layer perceptron built using custom `PrunableLinear` layers
- Optimizer: AdamW
- Metric 1: Test Accuracy 
- Metric 2: Sparsity Level (%), measured as the percentage of gate values below `1e-2` 
- Compared λ values: `0.0`, `1e-5`, `5e-5`, `1e-4` 
Results

| Lambda | Test Accuracy | Sparsity Level (%) |
|-------:|--------------:|-------------------:|
| 0.0    | 60.22         | 0.00               |
| 1e-5   | 59.73         | 1.11               |
| 5e-5   | 59.94         | 35.12              |
| 1e-4   | 59.66         | 58.71              |

## Analysis
The results show the expected sparsity-versus-accuracy trade-off required by the case study. 
With `λ = 0`, the model focuses only on classification and produces almost no pruning, while larger values of `λ` increase sparsity by penalizing active gates more strongly.
