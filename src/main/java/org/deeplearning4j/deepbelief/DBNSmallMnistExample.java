package org.deeplearning4j.deepbelief;


import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.plot.iterationlistener.NeuralNetPlotterIterationListener;
import org.deeplearning4j.plot.iterationlistener.PlotFiltersIterationListener;
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
public class DBNSmallMnistExample {

    private static Logger log = LoggerFactory.getLogger(DBNSmallMnistExample.class);


    public static void main(String[] args) throws Exception {
        Nd4j.MAX_SLICES_TO_PRINT = 10;
        Nd4j.MAX_ELEMENTS_PER_SLICE = 40;

        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10;
        int numSamples = 1000;
        int batchSize = 1000;
        int iterations = 100;
        int seed = 123;
        int listenerFreq = 1;

        log.info("Load data....");
        DataSetIterator iter = new MnistDataSetIterator(batchSize,numSamples);

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(numRows * numColumns)
                .nOut(outputNum)
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                .activationFunction("sigmoid")
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .iterations(iterations).l1(1e-1).regularization(true).constrainGradientToUnitNorm(true)
                .learningRate(1e-1f).updater(Updater.ADADELTA)
                .list(3)
                .hiddenLayerSizes(600, 400, 200)
                .override(2, new ClassifierOverride() {
                    @Override
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        builder.activationFunction("softmax");
                        builder.layer(new OutputLayer());
                        builder.lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD);
                    }
                })
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Arrays.asList(new ScoreIterationListener(listenerFreq), new NeuralNetPlotterIterationListener(50)));

        log.info("Train model....");
        model.fit(iter.next());
        iter.reset();



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
