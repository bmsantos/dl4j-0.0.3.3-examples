package org.deeplearning4j.rbm;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
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

import java.util.*;

public class RBMCreateDataExample {

    private static Logger log = LoggerFactory.getLogger(RBMCreateDataExample.class);

    public static void main(String... args) throws Exception {
        int numFeatures = 40;
        int iterations = 5;
        int seed = 123;
        int listenerFreq = iterations/5;
        Nd4j.getRandom().setSeed(seed);
        Nd4j.MAX_ELEMENTS_PER_SLICE = Integer.MAX_VALUE;
        Nd4j.MAX_SLICES_TO_PRINT = Integer.MAX_VALUE;
        log.info("Load dat....");
        INDArray input = Nd4j.create(2, numFeatures); // have to be at least two or else output layer gradient is a scalar and cause exception
        INDArray labels = Nd4j.create(2, 2);

        INDArray row0 = Nd4j.create(1, numFeatures);
        row0.assign(0.1);
        input.putRow(0, row0);
        labels.put(0, 1, 1); // set the 4th column

        INDArray row1 = Nd4j.create(1, numFeatures);
        row1.assign(0.2);

        input.putRow(1, row1);
        labels.put(1, 0, 1); // set the 2nd column

        DataSet trainingSet = new DataSet(input, labels);


        log.info("Build model....");
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(trainingSet.numInputs())
                .nOut(trainingSet.numOutcomes())
                .seed(seed)
                .weightInit(WeightInit.SIZE)
                .constrainGradientToUnitNorm(true)
                .iterations(iterations)
                .activationFunction("tanh")
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .build();
        Layer model = LayerFactories.getFactory(conf).create(conf);
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));

        log.info("Evaluate weights....");
        INDArray w = model.getParam(DefaultParamInitializer.WEIGHT_KEY);
        log.info("Weights: " + w);

        log.info("Train model....");
        model.fit(trainingSet.getFeatureMatrix());


    }

    // A single layer just learns features and is not supervised learning.

}