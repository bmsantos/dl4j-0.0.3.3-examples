package org.deeplearning4j.deepbelief;


import org.deeplearning4j.datasets.fetchers.LFWDataFetcher;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;


/**
 * Created by agibsonccc on 10/2/14.
 **/
public class DBNLFWExample {
    private static Logger log = LoggerFactory.getLogger(DBNLFWExample.class);


    public static void main(String[] args) throws Exception {

        int numSamples = LFWDataFetcher.NUM_IMAGES;
        int batchSize = 1000;
        int iterations = 5;
        int seed = 123;
        int rows = 28;
        int columns = 28;
        int listenerFreq = iterations/5;

        log.info("Load data....");
        DataSetIterator dataIter = new LFWDataSetIterator(batchSize,numSamples,rows,columns);

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .constrainGradientToUnitNorm(true)
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .list(4)
                .layer(0, new RBM.Builder().nIn(dataIter.inputColumns()).nOut(600).build())
                .layer(1, new RBM.Builder().nIn(600).nOut(250).build())
                .layer(2, new RBM.Builder().nIn(250).nOut(200).build())
                .layer(3, new OutputLayer.Builder(LossFunction.RMSE_XENT).activation("softmax")
                	.nIn(200).nOut(dataIter.totalOutcomes()).build())
            	.pretrain(true).backward(false)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        log.info("Train model....");
        while(dataIter.hasNext()) {
            DataSet next = dataIter.next();
            next.normalizeZeroMeanZeroUnitVariance();
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
