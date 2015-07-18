
CoDL4J Examples 
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

| **Model Name**      | **Accuracy** | **F1** | **Status**   | **Training**  |
|---------------------|--------------|--------|--------------|---------------|
| CNNIris             | 0.75         | 0.736  | Tune         | batch         |
| CNNMnist            | 0.155        | 0.011  | Fix          | batch         | - only predicts 0
| CNNMnist2           | 0.05         | 0.009  | Fix          | batch         | - only predicts 0
| DBNCreateData       | 0.50         | 0.33   | Fix          | batch         | - predicts NAN
| DBNFullMnist        | 0.357        | 0.018  | Tune         | full          | - only predicts 0
| DBNIris             | 0.975        | 0.962  | Tune             | full          | 
| DBNLWF              | 5.0E-3       | 3.8E-4 | Tune         | batch         | - only predicts 0
| DBNMnistReconstruct | 0.347        | 0.017  | Tune         | batch         | - only predicts 0
| DBNSmallMnist       | 0.425        | 0.023  | Fix          | full          | - only predicts 0
| GloveRawSentence    | Sim 0.13     | NA     | Tune         | batch         |
| MLPBackpropIris     | 0.609        | 0.513  | Tune         | batch         | 
| RBMCreateData	      |              | NA     | Fix          | full          |
| RBMIris             |              | NA     | Tune         | full          |
| RecurrentLSTMMnist  |              | NA     | Validate     | batch         |
| RecursiveAutoEncoder|              | NA     | Validate     | batch         |
| RNTNTweets          |              | 0.33   | Fix          | batch         | - predicts multiple
| RNTNTweets2         |              | 0.33   | Fix          | batch         | - predicts multiple
| TSNEBarnesHut       |              | NA     | Fix          | NA            |
| TSNEStandard        |              | NA     | Fix          | NA            |
| Word2VecRawText     | Sim 0.24     | NA     | Fix          | batch         |
    

*Sim is simularity
**F1 scores vary with each run. Seed has been added to some examples to stabilize results. Other examples need adjustments for seed to work.

Running MNIST is significantly slow.
