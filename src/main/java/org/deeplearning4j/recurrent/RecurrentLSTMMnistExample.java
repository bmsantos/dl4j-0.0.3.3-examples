package org.deeplearning4j.recurrent;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
//import org.deeplearning4j.nn.layers.recurrent.LSTM;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import java.util.Arrays;
import java.util.Collections;

/**
 * Created by willow on 5/11/15.
 */

public class RecurrentLSTMMnistExample {

    private static Logger log = LoggerFactory.getLogger(RecurrentLSTMMnistExample.class);

    public static void main(String[] args) throws Exception {

        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10;
        int numSamples = 100;
        int batchSize = 100;
        int iterations = 100;
        int seed = 123;
        int listenerFreq = iterations/5;

        log.info("Loading data...");
        MnistDataFetcher fetcher = new MnistDataFetcher(true);

        log.info("Building model...");
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new LSTM())
                .nIn(numRows * numColumns)
                .nOut(numRows * numColumns)
                .activationFunction("sigmoid")
                .optimizationAlgo(OptimizationAlgorithm.LBFGS)
                .constrainGradientToUnitNorm(true)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .build();
        Layer model = LayerFactories.getFactory(conf.getLayer()).create(conf);
        model.setIterationListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));

        log.info("Training model...");
        for(int i=0 ; i < (numSamples/batchSize); i++) {
            fetcher.fetch(batchSize);
            DataSet mnist = fetcher.next();
            model.fit(mnist.getFeatureMatrix());
        }

        // TODO add listener for graphs
        // Generative model - unsupervised and its time series based which requires different evaluation technique

    }

}
