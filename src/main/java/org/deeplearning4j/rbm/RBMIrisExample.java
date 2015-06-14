package org.deeplearning4j.rbm;


import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;


/**
 * Created by agibsonccc on 9/12/14.
 *
 * ? Output layer not a instance of output layer returning ?
 *
 */
public class RBMIrisExample {

    private static Logger log = LoggerFactory.getLogger(RBMIrisExample.class);

    public static void main(String[] args) throws IOException {
        // Customizing params
        Nd4j.MAX_SLICES_TO_PRINT = -1;
        Nd4j.MAX_ELEMENTS_PER_SLICE = -1;

        log.info("Load data....");
        DataSetIterator iter = new IrisDataSetIterator(150, 150); 
        DataSet iris = iter.next(); //DataSetIterator loads data from file into 
                                    //DataSet structure that neural net can use. 
        iris.normalizeZeroMeanZeroUnitVariance();

        log.info("Build model....");
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM()) //we define the layer as an RBM
                .nIn(4) // no. inputs = no. of Iris features
                .nOut(3) // no. outputs = no. of Iris species/labels
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN) //add Gaussian noise to generalize
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED) //Relu transform
                .iterations(100) // run 100 times to train
                .weightInit(WeightInit.DISTRIBUTION) //rand. initialization of weights
                .dist(new UniformDistribution(0, 1)) // mean of 0, st. dev. of 1
                .activationFunction("tanh") //squash the output w/nonlinear sigmoid transform tanh
                .k(1) // k = no. samples to collect for 1 iteration. k=1 is normal
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT) //root-mean-squared-error cross entropy to measure error
                .learningRate(1e-1f) //step size adjusting weights at each iteration
                .momentum(0.9) //2nd-order coefficient for the learning rate
                .regularization(true) // regularization fights overfitting.
                .l2(2e-4) // L2 regularization punishes high values in coefficients
                .optimizationAlgo(OptimizationAlgorithm.LBFGS) // calculates gradients along which params are optimized
                .constrainGradientToUnitNorm(true) 
                .build();
        Layer model = LayerFactories.getFactory(conf.getLayer()).create(conf);
        model.setIterationListeners(Arrays.asList((IterationListener) new ScoreIterationListener(1)));

        log.info("Train model....");
        model.fit(iris.getFeatureMatrix()); // calling fit initializes the model's learning process.

        // A single layer just learns features and can't be supervised. Thus it cannot be evaluated.
    }
}
