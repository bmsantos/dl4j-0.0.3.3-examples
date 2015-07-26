package org.deeplearning4j.logistic;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * @author Adam Gibson
 */
public class LogisticRegressionMnist {

    private static Logger log = LoggerFactory.getLogger(LogisticRegressionMnist.class);

    public static void main(String[] args) {
        NeuralNetConfiguration neuralNetConfiguration = new NeuralNetConfiguration.Builder()
                .lossFunction(LossFunctions.LossFunction.MCXENT).
                        optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .activationFunction("softmax")
                .regularization(true)
                .l1(1e-1).l2(1e-3).seed(123)
                .iterations(1000)
                .weightInit(WeightInit.XAVIER)
                .learningRate(1e-1)
                .nIn(4).nOut(3)
                .layer(new org.deeplearning4j.nn.conf.layers.OutputLayer()).build();

        OutputLayer o = LayerFactories.getFactory(neuralNetConfiguration).create(neuralNetConfiguration);

        int numSamples = 150;
        int batchSize = 150;


        DataSetIterator iter = new IrisDataSetIterator(batchSize, numSamples);
        DataSet iris = iter.next(); // Loads data into generator and format consumable for NN
        iris.scale();
        o.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(1)));
        SplitTestAndTrain t = iris.splitTestAndTrain(0.5);
        o.fit(t.getTrain());
        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(3);
        eval.eval(t.getTest().getLabels(),o.output(t.getTest().getFeatureMatrix(), true));
        log.info(eval.stats());



    }

}
