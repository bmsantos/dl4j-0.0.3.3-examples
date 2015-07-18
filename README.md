
DL4J Examples 
=========================
*based on version 0.0.3.3.\**

Repository of Deeplearning4J neural net examples:

- Convolutional Neural Nets
- Deep-belief Neural Nets
- Glove Example
- Restricted Boltzmann Machines
- Recurrent Neural Nets
- Recursive Neural Nets
- TSNE
- Word2Vec

---
## Development
We are still developing and tuning these examples. If you notice issues, please log them, and if you want to contribute, submit a pull request. Input is welcome here.

Check the POM to confirm where these examples are running from. If it has SNAPSHOT in the dl4j and nd4j then *git clone* those repositories and build locally. Otherwise use Maven Repo. We are working to get this stabilized as quickly as possible.

## Documentation
For more information, check out [deeplearning4j.org](http://deeplearning4j.org/) and its [JavaDoc](http://deeplearning4j.org/doc/).

## Performance

| **Model Name**      | **Accuracy** | **F1** | **Status**   | **Training**  |**Timing**|
|---------------------|--------------|--------|--------------|---------------|----------|
| CNNIris             | 0.48         | 0.19   | Fix          | full          |          | 
| CNNMnist            | 0.34         | 0.032  | Fix          | full          |          | 
| CNNMnist2           | 0.05         | 0.009  | Fix          | batch         |          | 
| DBNCreateData       | 0.50         | 0.66   | Fix          | batch         |          | 
| DBNFullMnist        | 0.39         | 0.20   | Tune         | batch         | 63qm7.25s | - only predicts 1
| DBNIris             | 0.83         | 0.88   | Tune         | full          | 0m3.78s  | - with listeners on
| DBNLFW              |              |        | Check        | batch         |          |
| DBNMnistSingleLayer | 0.35         | 0.18   | Tune         | full          | 0m0.08s  | - only 500 examples
| DBNSmallMnist       | 0.54         | 0.29   | Tune         | full          | 0m0.09s  | - only 100 examples
| GloveRawSentence    |              | NA     |              | batch         | 0m0.73s  |
| MLPBackpropIris     | 0.42         | 0.54   | Tune         | batch         | 0m0.12s  |
| RBMCreateData	      |              | NA     | Tune         | full          | 0m0.09s  | - very small sample
| RBMIris             |              | NA     | Tune         | full          | 0m6.12s  |
| RecurrentLSTMMnist  |              | NA     | Fix          | batch         |          |
| RecursiveAutoEncoder|              | NA     | Validate     | batch         |          |
| RNTNTweets          |              | 0.33   | Fix          | batch         |          |
| RNTNTweets2         |              | 0.33   | Fix          | batch         |          |
| TSNEBarnesHut       |              | NA     | Fix          | NA            |          |
| TSNEStandard        |              | NA     | Fix          | NA            |          |
| Word2VecRawText     |              | NA     | Fix          | batch         | 0m1.36s  |
    

* using bash real for timing and running on single CPU with 2.3GHz processor
* Some networks need adjustments for seed to work (e.g. RNTN)
