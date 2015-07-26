package org.deeplearning4j.multilayer;


import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.AutoEncoder;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
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
import java.util.Collections;


/**
 * Created by agibsonccc on 9/12/14.
 *
 * ? Output layer not a instance of output layer returning ?
 *
 */
public class MLPBackpropIrisExample {

    private static Logger log = LoggerFactory.getLogger(MLPBackpropIrisExample.class);

    public static void main(String[] args) throws IOException {
        // Customizing params
        Nd4j.MAX_SLICES_TO_PRINT = 10;
        Nd4j.MAX_ELEMENTS_PER_SLICE = 10;

        final int numInputs = 4;
        int outputNum = 3;
        int numSamples = 150;
        int batchSize = 150;
        int iterations = 100;
        long seed = 6;
        int listenerFreq = 1;
        log.info("Load data....");
        DataSetIterator iter = new IrisDataSetIterator(batchSize, numSamples);

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(numInputs)
                .nOut(outputNum).seed(123)
                .seed(seed).regularization(true).l2(1e-1).l1(1e-3)
                .iterations(iterations).dropOut(0.0).constrainGradientToUnitNorm(true)
                .weightInit(WeightInit.XAVIER).corruptionLevel(0.6)
                .activationFunction("relu").updater(Updater.ADAM)
                .learningRate(1e-6)
                .list(3)
                .backward(true)
                .pretrain(false)
                .hiddenLayerSizes(3,2,2)
                .override(3, new ConfOverride() {
                    @Override
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        builder.activationFunction("softmax");
                        builder.layer(new OutputLayer());
                        builder.lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD);
                    }
                }).build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));

        log.info("Train model....");
        DataSet iris = iter.next();

        iris.normalizeZeroMeanZeroUnitVariance();
        SplitTestAndTrain testAndTrain = iris.splitTestAndTrain(0.8);
        model.fit(testAndTrain.getTrain());
        iter.reset();

        log.info("Evaluate weights....");
        for(org.deeplearning4j.nn.api.Layer layer : model.getLayers()) {
            INDArray w = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
            log.info("Weights: " + w);
        }


        log.info("Evaluate model....");
        Evaluation eval = new Evaluation();
        INDArray output = model.output(testAndTrain.getTest().getFeatureMatrix(),true);
        eval.eval(testAndTrain.getTest().getLabels(), output);
        log.info(eval.stats());
        log.info("****************Example finished********************");

    }
}
