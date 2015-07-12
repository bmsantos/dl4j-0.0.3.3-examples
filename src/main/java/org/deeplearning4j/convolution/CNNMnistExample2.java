package org.deeplearning4j.convolution;

import com.fasterxml.jackson.core.sym.Name3;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.conf.rng.DefaultRandom;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionInputPreProcessor;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionPostProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.ArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Created by willow on 5/11/15.
 */
public class CNNMnistExample2 {

    private static final Logger log = LoggerFactory.getLogger(CNNMnistExample2.class);
    private static final int numRows = 28;
    private static final int numColumns = 28;
    private static final int outputNum = 10;

    @Option(name="-samples", usage="number of samples to get")
    private static int numSamples = 100;

    @Option(name="-batch", usage="batch size for training" )
    private static int batchSize = 100;

    @Option(name="-iterations", usage="number of iterations to train the layer")
    private static int iterations = 10;

    @Option(name="-featureMap", usage="size of feature map. Just enter single value")
    int featureMapSize = 9;

    @Option(name="-learningRate", usage="learning rate")
    private static double learningRate = 0.13;

    @Option(name="-hLayerSize", usage="hidden layer size")
    private static int hLayerSize = 50;

    private static int splitTrainNum = (int) (numSamples * 0.8);
    private static int numTestSamples = numSamples - splitTrainNum;
    private static int listenerFreq = iterations/5;


    static DataSetIterator loadData(int batchSize, int numTrainSamples) throws Exception{
        //SamplingDataSetIterator - TODO make sure representation of each classification in each batch
        DataSetIterator dataIter = new MnistDataSetIterator(batchSize, numTrainSamples);
        return dataIter;
    }

    static MultiLayerNetwork buildModel(final int featureMapSize){
        // Uniform and Zero have good results
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .nIn(numRows * numColumns)
                .nOut(outputNum)
                .seed(123)
                .batchSize(batchSize)
                .iterations(iterations)
                .weightInit(WeightInit.SIZE)
                .activationFunction("relu")
                .filterSize(8, 1, numRows, numColumns)
                .learningRate(learningRate)
                .optimizationAlgo(OptimizationAlgorithm.LBFGS)
                .constrainGradientToUnitNorm(true)
                .list(3)
                .hiddenLayerSizes(hLayerSize)
                .inputPreProcessor(0, new ConvolutionInputPreProcessor(numRows, numColumns))
                .preProcessor(1, new ConvolutionPostProcessor())
                .override(0, new ConfOverride() {
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        builder.layer(new ConvolutionLayer());
                        builder.convolutionType(ConvolutionLayer.ConvolutionType.MAX);
                        builder.featureMapSize(featureMapSize, featureMapSize);
                    }
                })
                .override(1, new ConfOverride() {
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        builder.layer(new SubsamplingLayer());
                    }
                })
                .override(2, new ClassifierOverride() {
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        builder.layer(new OutputLayer());
                        builder.activationFunction("softmax");
                        builder.optimizationAlgo(OptimizationAlgorithm.GRADIENT_DESCENT);
                        builder.lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD);
                    }
                })
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));
        return model;
    }

    static MultiLayerNetwork trainModel(DataSetIterator data, MultiLayerNetwork model){
        while (data.hasNext()){
            DataSet allData = data.next();
            allData.normalizeZeroMeanZeroUnitVariance();
            model.fit(allData);
        }
        return model;
    }

    static void evaluateModel(DataSetIterator data, MultiLayerNetwork model) {
        data.reset();

        Evaluation eval = new Evaluation();
        DataSet allTest = data.next();
        INDArray testInput = allTest.getFeatureMatrix();
        INDArray testLabels = allTest.getLabels();
        INDArray output = model.output(testInput);

//        log.info("DATA************");
//        log.info(testLabels.toString());
//        log.info(output.toString());

        eval.eval(testLabels, output);
        log.info(eval.stats());

    }

    public void exec(String[] args) throws Exception {
        Nd4j.MAX_SLICES_TO_PRINT = 10;
        Nd4j.MAX_ELEMENTS_PER_SLICE = 10;
        Nd4j.MAX_ELEMENTS_PER_SLICE = 10;

        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);

        } catch (CmdLineException e) {
            // handling of wrong arguments
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
        }

        log.info("Load data....");
        DataSetIterator dataIter = loadData(batchSize, splitTrainNum);

        log.info("Build model....");
        MultiLayerNetwork model = buildModel(featureMapSize);

        log.info("Train model....");
        model = trainModel(dataIter, model);

        log.info("Evaluate model....");
        DataSetIterator testData = loadData((int) numTestSamples, numTestSamples);
        evaluateModel(testData, model);

        log.info("****************Example finished********************");
    }

    public static void main( String[] args ) throws Exception
    {
        new CNNMnistExample2().exec(args);

    }
}