package org.deeplearning4j.convolution;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.layers.convolution.ConvolutionDownSampleLayer;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionInputPreProcessor;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionPostProcessor;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
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

import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

/**
 * @author sonali
 */
public class CNNIrisExample {

    private static Logger log = LoggerFactory.getLogger(CNNIrisExample.class);

    public static void main(String[] args) {

        final int numRows = 2;
        final int numColumns = 2;
        int outputNum = 3;
        int numSamples = 150;
        int batchSize = 150;
        int iterations = 10;
        int splitTrainNum = 110;
        int seed = 123;
        int listenerFreq = iterations / 5;


        /**
         *Set a neural network configuration with multiple layers
         */
        log.info("Load data....");
        DataSetIterator irisIter = new IrisDataSetIterator(batchSize, numSamples);
        DataSet iris = irisIter.next();
        iris.normalizeZeroMeanZeroUnitVariance();

        SplitTestAndTrain trainTest = iris.splitTestAndTrain(splitTrainNum, new Random(seed));

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .nIn(numRows * numColumns)
                .nOut(outputNum)
                .seed(seed)
                .iterations(iterations)
                .weightInit(WeightInit.XAVIER)
                .activationFunction("relu")
                .filterSize(5, 1, numRows, numColumns)
                .batchSize(batchSize)
                .optimizationAlgo(OptimizationAlgorithm.LBFGS)
                .constrainGradientToUnitNorm(true)
                .regularization(true)
                .l2(2e-4)
                .list(2)
                .hiddenLayerSizes(4)
                .inputPreProcessor(0, new ConvolutionInputPreProcessor(numRows, numColumns))
                .preProcessor(0, new ConvolutionPostProcessor())
                .useDropConnect(true)
                .override(0, new ConfOverride() {
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        builder.layer(new ConvolutionLayer());
                        builder.convolutionType(ConvolutionLayer.ConvolutionType.MAX);
                        builder.featureMapSize(2, 2);
                    }
                })
                .override(1, new ConfOverride() {
                    @Override
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        builder.activationFunction("softmax");
                        builder.layer(new OutputLayer());
                        builder.lossFunction(LossFunctions.LossFunction.MCXENT);
                    }
                })
                .build();

        log.info("Build model....");
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq));

        log.info("Train model....");
        model.fit(trainTest.getTrain());

        log.info("Evaluate weights....");
        for(org.deeplearning4j.nn.api.Layer layer : model.getLayers()) {
            INDArray w = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
            log.info("Weights: " + w);
        }

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation();
        INDArray output = model.output(trainTest.getTest().getFeatureMatrix());
        eval.eval(trainTest.getTest().getLabels(), output);
        log.info(eval.stats());

        log.info("****************Example finished********************");
    }
}
