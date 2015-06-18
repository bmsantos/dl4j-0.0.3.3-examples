package org.deeplearning4j.deepbelief;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
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
import java.util.Collections;


/**
 * Created by willow on 5/22/15.
 */
public class DBNCreateDataExample {


    private static Logger log = LoggerFactory.getLogger(DBNCreateDataExample.class);

    public static void main(String... args) throws Exception {
        int numFeatures = 614;
        int iterations = 5;
        int seed = 123;
        int listenerFreq = iterations/5;


        log.info("Load data....");
        // have to be at least two or else output layer gradient is a scalar and cause exception
        INDArray input = Nd4j.create(2, numFeatures);
        INDArray labels = Nd4j.create(2, 2);

        INDArray row0 = Nd4j.create(1, numFeatures);
        row0.assign(0.1);
        input.putRow(0, row0);
        labels.put(0, 1, 1); // set the 1st column

        INDArray row1 = Nd4j.create(1, numFeatures);
        row1.assign(0.2);
        input.putRow(1, row1);
        labels.put(1, 0, 1); // set the 2nd column

        DataSet trainingSet = new DataSet(input, labels);

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(trainingSet.numInputs())
                .nOut(trainingSet.numOutcomes())
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0,1))
                .seed(seed)
                .constrainGradientToUnitNorm(true)
                .iterations(iterations)
                .activationFunction("tanh")
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-2f)
                .optimizationAlgo(OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT)
                .list(2)
                .hiddenLayerSizes(400)
                .override(1, new ClassifierOverride())
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));

        log.info("Train model....");
        model.fit(trainingSet);

        log.info("Evaluate weights....");
        for(org.deeplearning4j.nn.api.Layer layer : model.getLayers()) {
            INDArray w = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
            log.info("Weights: " + w);
        }

        log.info("Evaluate model....");
        INDArray predictedMatrix = model.output(trainingSet.getFeatureMatrix());
        for (int i = 0; i < predictedMatrix.rows(); i++) {
            String actual = trainingSet.getLabels().getRow(i).toString().trim();
            String predicted = predictedMatrix.getRow(i).toString().trim();
            log.info("actual " + actual + " vs predicted " + predicted);
        }
        Evaluation eval = new Evaluation();
        eval.eval(trainingSet.getLabels(), predictedMatrix);
        log.info(eval.stats());

        log.info("****************Example finished********************");
    }
}