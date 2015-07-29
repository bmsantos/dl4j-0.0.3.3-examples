package org.deeplearning4j.rbm;


import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.layers.RBM.*;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;


/**
 * @author Yoda the Jedi master
 *
 */
public class RBMIrisExample {

    private static Logger log = LoggerFactory.getLogger(RBMIrisExample.class);

    public static void main(String[] args) throws IOException {
        // Customizing params
        Nd4j.MAX_SLICES_TO_PRINT = -1;
        Nd4j.MAX_ELEMENTS_PER_SLICE = -1;

        final int numRows = 4;
        final int numColumns = 1;
        int outputNum = 3;
        int numSamples = 150;
        int batchSize = 150;
        int iterations = 100;
        int seed = 123;
        int listenerFreq = iterations/5;

        log.info("Load data....");
        DataSetIterator iter = new IrisDataSetIterator(batchSize, numSamples);
        DataSet iris = iter.next(); // Loads data into generator and format consumable for NN

        iris.normalizeZeroMeanZeroUnitVariance();

        log.info("Build model....");
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                // Gaussian for the visible layer and Rectify for the hidden layer
                // Set contrastive divergence to 1
                .layer(new RBM.Builder(HiddenUnit.RECTIFIED, VisibleUnit.GAUSSIAN, 1)
                        // Input nodes
                        .nIn(numRows * numColumns)
                        // Output nodes
                        .nOut(outputNum)
                        // Activation function type
                        .activation("tanh")
                        .build())
                .seed(seed) // Seed to lock in weight initialization for tuning
                .weightInit(WeightInit.DISTRIBUTION) // Weight initialization method
                .dist(new UniformDistribution(0, 1))  // Weight distribution curve mean and stdev
                .lossFunction(LossFunctions.LossFunction.SQUARED_LOSS) // Loss function type
                .learningRate(1e-1f) // Backprop step size
                .momentum(0.9) // Speed of modifying learning rate
                .regularization(true) // Prevent overfitting
                .l2(2e-4) // Regularization type
                .optimizationAlgo(OptimizationAlgorithm.LBFGS) // Backprop method (calculate the gradients)
                .constrainGradientToUnitNorm(true)
                .build();
        Layer model = LayerFactories.getFactory(conf.getLayer()).create(conf);
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        log.info("Evaluate weights....");
        INDArray w = model.getParam(DefaultParamInitializer.WEIGHT_KEY);
        log.info("Weights: " + w);

        log.info("Train model....");
        model.fit(iris.getFeatureMatrix());

        log.info("Visualize training results....");
        NeuralNetPlotter plotter = new NeuralNetPlotter();
        plotter.plotNetworkGradient(model, model.gradient());
    }


    // A single layer just learns features and is not supervised learning.

}
