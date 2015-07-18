package org.deeplearning4j.deepbelief;


import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.SamplingDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.plot.iterationlistener.ActivationMeanIterationListener;
import org.deeplearning4j.ui.renders.UpdateFilterIterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Collections;

/**
 * Created by agibsonccc on 9/11/14.
 */
public class DBNFullMnistExample {

    private static Logger log = LoggerFactory.getLogger(DBNFullMnistExample.class);

    public static void main(String[] args) throws Exception {
        Nd4j.MAX_ELEMENTS_PER_SLICE = Integer.MAX_VALUE;
        Nd4j.MAX_SLICES_TO_PRINT = Integer.MAX_VALUE;
        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10;
        int numSamples = 60000;
        int batchSize = 1000;
        int iterations = 10;
        int seed = 123;
        int listenerFreq = batchSize / 5;

        log.info("Load data....");

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(numRows * numColumns)
                .nOut(outputNum)
                .weightInit(WeightInit.XAVIER)
                .seed(seed)
                .constrainGradientToUnitNorm(true)
                .iterations(iterations).activationFunction("sigmoid")
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f).updater(Updater.ADAM)
                .momentumAfter(Collections.singletonMap(3, 0.9))
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .list(4)
                .hiddenLayerSizes(600, 250, 200)
                .override(3, new ClassifierOverride())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Arrays.asList(new ScoreIterationListener(listenerFreq),new HistogramIterationListener(1)
                ,new ActivationMeanIterationListener(1),new UpdateFilterIterationListener(Arrays.asList(PretrainParamInitializer.WEIGHT_KEY),1)));

        log.info("Train model....");
        //model.fit(iter); // achieves end to end pre-training

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation();
        DataSet all = new MnistDataSetIterator(60000,60000).next();
        all.shuffle();
        DataSetIterator testIter = new MultipleEpochsIterator(3,new SamplingDataSetIterator(all,10,600000));
        model.fit(testIter);

        log.info(eval.stats());
        log.info("****************Example finished********************");

    }

}
