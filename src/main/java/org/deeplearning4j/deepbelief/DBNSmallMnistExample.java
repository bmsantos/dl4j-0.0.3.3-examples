package org.deeplearning4j.deepbelief;


import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.plot.iterationlistener.ActivationMeanIterationListener;
import org.deeplearning4j.plot.iterationlistener.NeuralNetPlotterIterationListener;
import org.deeplearning4j.plot.iterationlistener.PlotFiltersIterationListener;
import org.deeplearning4j.ui.activation.UpdateActivationIterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
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

        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10;
        int numSamples = 100;
        int batchSize = 100;
        int iterations = 5;
        int seed = 123;
        int listenerFreq = iterations/5;

        log.info("Load data....");
        DataSetIterator iter = new MultipleEpochsIterator(5, new MnistDataSetIterator(batchSize,numSamples));

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(numRows * numColumns)
                .nOut(outputNum)
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .constrainGradientToUnitNorm(true)
                .iterations(iterations)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f)
                .list(4)
                .hiddenLayerSizes(600, 500, 400)
                .override(3, new ConfOverride() {
                    @Override
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        builder.activationFunction("softmax");
                        builder.layer(new OutputLayer());
                        builder.lossFunction(LossFunctions.LossFunction.MCXENT);
                        builder.optimizationAlgo(OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT);
                    }
                })
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
