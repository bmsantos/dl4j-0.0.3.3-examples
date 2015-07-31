package org.deeplearning4j.deepbelief;


import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
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
import java.util.Random;


/**
 * Created by agibsonccc on 9/12/14.
 *
 */
public class DBNIrisExample {

    private static Logger log = LoggerFactory.getLogger(DBNIrisExample.class);

    public static void main(String[] args) throws IOException {
        // Customizing params
        Nd4j.MAX_SLICES_TO_PRINT = -1;
        Nd4j.MAX_ELEMENTS_PER_SLICE = -1;

        final int numRows = 4;
        final int numColumns = 1;
        int outputNum = 3;
        int numSamples = 150;
        int batchSize = 150;
        int iterations = 2;
        int splitTrainNum = (int) (batchSize * .8);
        int seed = 123;
        int listenerFreq = 1;

        log.info("Load data....");
        DataSetIterator iter = new IrisDataSetIterator(batchSize, numSamples);
        DataSet next = iter.next();
        next.scale();

        log.info("Split data....");
        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(splitTrainNum, new Random(seed));
        DataSet train = testAndTrain.getTrain();
        DataSet test = testAndTrain.getTest();
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed) // Seed to lock in weight initialization for tuning
                .iterations(iterations) // # training iterations predict/classify & backprop
                .weightInit(WeightInit.XAVIER) // Weight initialization method
                .activationFunction("relu") // Activation function type
                .k(1) // # contrastive divergence iterations
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT) // Loss function type
                .learningRate(1e-3f) // Optimization step size
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT) // Backprop method (calculate the gradients)
                .momentum(0.9)
                .updater(Updater.ADAGRAD)
                .constrainGradientToUnitNorm(true)
                .dropOut(0.5)
                .useDropConnect(true)
                .list(2) // # NN layers (does not count input layer)
                .layer(0, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
                                .nIn(numRows * numColumns) // # input nodes
                                .nOut(3) // # fully connected hidden layer nodes. Add list if multiple layers.
                                .build()
                ) // NN layer type
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                .nIn(3) // # input nodes
                                .nOut(outputNum) // # output nodes
                                .activation("softmax")
                                .build()
                ) // NN layer type
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
//        model.setListeners(Arrays.asList(new ScoreIterationListener(listenerFreq),
//                new GradientPlotterIterationListener(listenerFreq),
//                new LossPlotterIterationListener(listenerFreq)));


        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));
        log.info("Train model....");
        model.fit(train);

        log.info("Evaluate weights....");
        for(org.deeplearning4j.nn.api.Layer layer : model.getLayers()) {
            INDArray w = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
            log.info("Weights: " + w);
        }

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation();
        INDArray output = model.output(test.getFeatureMatrix());

        for (int i = 0; i < output.rows(); i++) {
            String actual = train.getLabels().getRow(i).toString().trim();
            String predicted = output.getRow(i).toString().trim();
            log.info("actual " + actual + " vs predicted " + predicted);
        }

        eval.eval(test.getLabels(), output);
        log.info(eval.stats());
        log.info("****************Example finished********************");

    }
}
