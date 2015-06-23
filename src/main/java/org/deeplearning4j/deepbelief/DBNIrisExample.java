package org.deeplearning4j.deepbelief;


import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.conf.rng.DefaultRandom;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
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
 */
public class DBNIrisExample {

    private static Logger log = LoggerFactory.getLogger(DBNIrisExample.class);

    public static void main(String[] args) throws IOException {
        // Customizing params
        Nd4j.MAX_SLICES_TO_PRINT = -1;
        Nd4j.MAX_ELEMENTS_PER_SLICE = -1;

        Nd4j.getRandom().setSeed(123);

        log.info("Load data....");
        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        DataSet next = iter.next();
        next.normalizeZeroMeanZeroUnitVariance();

        log.info("Split data....");
        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(110);
        DataSet train = testAndTrain.getTrain();
        DataSet test = testAndTrain.getTest();

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM()) //the nn's layers will be RBMs
                .nIn(4) // no. of Input nodes = 4
                .nOut(3) // no. of Output nodes/labels = 3
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN) //Gaussian transform
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED) // Rect. Linear trans.
                .iterations(100) // make 100 passes of guess and backprop
                .weightInit(WeightInit.VI) // initializes weights
                .activationFunction("relu") // sigmoid activation of nodes
                .k(1) // no. of times you run contrastive divergence
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                // your loss function = root-mean-squared error cross entropy
                .learningRate(1e-1) //the size of the steps your algo takes
                .momentum(0.9) //a coefficient that modifies the learning rate
                .regularization(true) // regularization fights overfitting
                .l2(2e-4) // l2 is one type of regularization
                .optimizationAlgo(OptimizationAlgorithm.LBFGS)
                //optimization algorithms calculate the gradients. 
                //LBFGS is one type.
                .constrainGradientToUnitNorm(true)
                .list(2)
                .hiddenLayerSizes(3) // no. of nodes in your hidden layer. 
                // this is small.
                .override(1, new ClassifierOverride())
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        Collections.singletonList((IterationListener) new ScoreIterationListener(1));

        log.info("Train model....");
        model.fit(train);

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
