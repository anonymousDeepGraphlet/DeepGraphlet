# DeepGraphlet
This project implements the DeepGraphlet model, which focuses on computing the local graphlet frequencies approximately by GNN models.

### Run the experiments

All the baselines, DeepGraphlet and its variants are contained in directory `source`. We can run the function implemented in `run.py` to run the experiments.

### Compile the code

Before running the model, we recommend switching the working directory to the `KtupleFeature`, then run command:

```
make -f new.make
```

### Input Format 

The input graph data are expected as below:

```markdown
The graph is consist of N vertices and M edges: 

N M

M edges:

x_1 y_1
x_2 y_2

```

### Time

|                    |                    LGC                    | LGC-multi task|   evoke    |
| :----------------: | :---------------------------------------: | :-----------: | :--------: |
|       artist       |          (19s + 27s + 44s)+0.6s           |      1.1s     |   1m57s    |
|       com-lj       |       (23m33s +35m51s + 1h27m)+45s        |      101s     |  5h28m44s  |
|    com-Berkstan    |       (3m59s + 6m6s + 14m38s)+5.7s        |       13s     |   59m57s   |
|        orkut       |     (25m46s + 37m39s + 57m37s) + 97s      |      198s     |  38h48m17s |
|     friendster     | (3h33m48s + 5h19m17s + 11h22m32s) + 4308s |     10830s    |    null    |
