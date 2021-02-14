# DeepGraphlet
This project implements the DeepGraphlet model, which focuses on computing the local graphlet frequencies approximately by GNN models.

### Run the experiments

All the baselines, DeepGraphlet and its variants are contained in directory `source`. We can run the function implemented in `run.py` to run the experiments.

### Compile the KtupleFeature code

Before running the model, we recommend switching the working directory to the `KtupleFeature`, then run command:

```
make -f new.make
```

### Ktuple feature Input Format 

The input graph data are expected as below:

```markdown
The graph is consist of N vertices and M edges: 

N M

M edges:

x_1 y_1
x_2 y_2
```
