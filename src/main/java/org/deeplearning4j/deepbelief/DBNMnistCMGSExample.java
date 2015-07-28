package org.deeplearning4j.deepbelief;


import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;


/**
 * Deep, Big, Simple Neural Nets Excel on Handwritten Digit Recognition
 * 2010 paper by Cireșan, Meier, Gambardella, and Schmidhuber
 * They achieved 99.65 accuracy
 */
public class DBNMnistCMGSExample {

    private static Logger log = LoggerFactory.getLogger(DBNMnistCMGSExample.class);


    public static void main(String[] args) throws Exception {
        Nd4j.MAX_SLICES_TO_PRINT = 10;
        Nd4j.MAX_ELEMENTS_PER_SLICE = 40;

        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10;
        int numSamples = 100;
        int batchSize = 100;
        int iterations = 50;
        int seed = 123;
        int listenerFreq = 10;

        log.info("Load data....");
//        DataSetIterator iter = new MultipleEpochsIterator(5, new MnistDataSetIterator(batchSize,numSamples));
        DataSetIterator iter = new MnistDataSetIterator(batchSize,numSamples);

        // TODO verify the configuration with paper
        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .weightInit(WeightInit.SIZE)
                .activationFunction("sigmoid")
                .lossFunction(LossFunctions.LossFunction.SQUARED_LOSS)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .constrainGradientToUnitNorm(true)
                .regularization(true)
                .learningRate(1e-6f)
                .list(6)
                .layer(0, new RBM.Builder()
                        .nIn(numRows * numColumns)
                        .nOut(2500)
                        .build())
                .layer(1, new RBM.Builder()
                        .nIn(2500)
                        .nOut(2000)
                        .build())
                .layer(2, new RBM.Builder()
                        .nIn(2000)
                        .nOut(1500)
                        .build())
                .layer(3, new RBM.Builder()
                        .nIn(1500)
                        .nOut(1000)
                        .build())
                .layer(4, new RBM.Builder()
                        .nIn(1000)
                        .nOut(500)
                        .build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(500)
                        .nOut(outputNum)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        log.info("Train model....");
        model.fit(iter);
        iter.reset();

        log.info("Evaluate weights....");
        for(org.deeplearning4j.nn.api.Layer layer : model.getLayers()) {
            INDArray w = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
            log.info("Weights: " + w);
        }

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation();
        while(iter.hasNext()) {
            DataSet test_data = iter.next();
            INDArray predict2 = model.output(test_data.getFeatureMatrix());
            eval.eval(test_data.getLabels(), predict2);
        }
        log.info(eval.stats());
        log.info("****************Example finished********************");

    }

}
