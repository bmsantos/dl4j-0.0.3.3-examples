package org.deeplearning4j.multilayer;


import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
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

        final int numRows = 4;
        final int numColumns = 1;
        int outputNum = 3;
        int numSamples = 150;
        int batchSize = 10;
        int iterations = 1000;
        int splitTrainNum = (int) (batchSize * .8);
        long seed = 123;
        int listenerFreq = iterations/5;

        log.info("Load data....");
        DataSetIterator iter = new IrisDataSetIterator(batchSize, numSamples);

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(numRows * numColumns)
                .nOut(outputNum)
                .seed(seed)
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .iterations(iterations)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new UniformDistribution(0, 1))
                .activationFunction("tanh")
                .k(1)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f)
                .momentum(0.9)
                .regularization(true)
                .l2(2e-4)
                .optimizationAlgo(OptimizationAlgorithm.LBFGS)
                .constrainGradientToUnitNorm(true)
                .list(2)
                .hiddenLayerSizes(new int[]{3})
                .backward(true)
                .override(1,new ClassifierOverride())
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));

        log.info("Train model....");
        while(iter.hasNext()) {
            DataSet iris = iter.next();
            iris.normalizeZeroMeanZeroUnitVariance();
            model.fit(iris);
        }
        iter.reset();

        log.info("Evaluate weights....");
        for(org.deeplearning4j.nn.api.Layer layer : model.getLayers()) {
            INDArray w = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
            log.info("Weights: " + w);
        }


        log.info("Evaluate model....");
        Evaluation eval = new Evaluation();
        DataSetIterator iterTest = new IrisDataSetIterator(numSamples, numSamples);
        DataSet test = iterTest.next();
        test.normalizeZeroMeanZeroUnitVariance();
        INDArray output = model.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(), output);
        log.info(eval.stats());
        log.info("****************Example finished********************");

    }
}
