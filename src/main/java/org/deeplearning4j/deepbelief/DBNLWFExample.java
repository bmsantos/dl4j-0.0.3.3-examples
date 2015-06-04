package org.deeplearning4j.deepbelief;


import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.plot.iterationlistener.NeuralNetPlotterIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Collections;


/**
 * Created by agibsonccc on 10/2/14.
 **/
public class DBNLWFExample {
    private static Logger log = LoggerFactory.getLogger(DBNLWFExample.class);


    public static void main(String[] args) throws Exception {

        log.info("Load data....");
        DataSetIterator dataIter = new LFWDataSetIterator(100,1000);

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(dataIter.inputColumns())
                .nOut(dataIter.totalOutcomes())
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(1e-3, 1e-1))
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .constrainGradientToUnitNorm(true)
                .learningRate(1e-3f)
                .list(4)
                .hiddenLayerSizes(600, 250, 200)
                .override(3,new ClassifierOverride())
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        Collections.singletonList((IterationListener) new ScoreIterationListener(1));

        log.info("Train model....");
        while(dataIter.hasNext()) {
            DataSet next = dataIter.next();
            next.scale();
            model.fit(next);
        }

        log.info("Evaluate model....");
        dataIter = new LFWDataSetIterator(100,100);
        DataSet dataSet = dataIter.next();
        Evaluation eval = new Evaluation();
        INDArray output = model.output(dataSet.getFeatureMatrix());
        eval.eval(dataSet.getLabels(), output);
        log.info(eval.stats());

    }


}
