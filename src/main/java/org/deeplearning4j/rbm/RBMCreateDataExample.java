package org.deeplearning4j.rbm;

import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.*;

public class RBMCreateDataExample {

    private static Logger log = LoggerFactory.getLogger(RBMCreateDataExample.class);

    protected static String writeMatrix(INDArray matrix) throws IOException {
        String filePath = System.getProperty("java.io.tmpdir") + File.separator + UUID.randomUUID().toString();
        File write = new File(filePath);
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(write,true));
        write.deleteOnExit();
        for(int i = 0; i < matrix.rows(); i++) {
            INDArray row = matrix.getRow(i);
            StringBuilder sb = new StringBuilder();
            for(int j = 0; j < row.length(); j++) {
                sb.append(String.format("%.10f", row.getDouble(j)));
                if(j < row.length() - 1)
                    sb.append(",");
            }
            sb.append("\n");
            String line = sb.toString();
            bos.write(line.getBytes());
            bos.flush();
        }

        bos.close();
        return filePath;
    }



    public static void main(String... args) throws Exception {
        int numFeatures = 40;

        log.info("Load data....");
        //        MersenneTwister gen = new MersenneTwister(123); // other data to test?

        INDArray input = Nd4j.create(2, numFeatures); // have to be at least two or else output layer gradient is a scalar and cause exception
        INDArray labels = Nd4j.create(2, 2);

        INDArray row0 = Nd4j.create(1, numFeatures);
        row0.assign(0.1);
        input.putRow(0, row0);
        labels.put(0, 1, 1); // set the 4th column

        INDArray row1 = Nd4j.create(1, numFeatures);
        row1.assign(0.2);

        input.putRow(1, row1);
        labels.put(1, 0, 1); // set the 2nd column

        DataSet trainingSet = new DataSet(input, labels);

        log.info("Build model....");
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(trainingSet.numInputs())
                .nOut(trainingSet.numOutcomes())
                .weightInit(WeightInit.SIZE)
                .constrainGradientToUnitNorm(true)
                .iterations(3)
                .activationFunction("tanh")
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f)
                .optimizationAlgo(OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT)
                .build();
        Layer model = LayerFactories.getFactory(conf).create(conf);
        Collections.singletonList((IterationListener) new ScoreIterationListener(1));

        log.info("Train model....");
        model.fit(trainingSet.getFeatureMatrix());

        ///////////////////////////
        Gradient gradient = model.gradient();
        log.info("PARAMS" + model.getParam("W"));
        log.info("GRADIENTS" + gradient);

        // Gradient titles - W, W-gradient, b, b-gradient, vb, vb-gradient
        Set<String> vars = new TreeSet<>(gradient.gradientForVariable().keySet()); // W, b, vb
        List<String> titles = new ArrayList<>(vars);
        for(String s : vars) {
            titles.add(s + "-gradient");
        }

        // Gradient matrices
        INDArray[] matrices = new INDArray[]{
                model.getParam(DefaultParamInitializer.WEIGHT_KEY),
                model.getParam(PretrainParamInitializer.BIAS_KEY),
                model.getParam(PretrainParamInitializer.VISIBLE_BIAS_KEY),
                gradient.gradientForVariable().get(DefaultParamInitializer.WEIGHT_KEY),
                gradient.gradientForVariable().get(DefaultParamInitializer.BIAS_KEY),
                gradient.gradientForVariable().get(PretrainParamInitializer.VISIBLE_BIAS_KEY)
        };

        String[] path = new String[titles.size() + matrices.length];
        for(int i = 0; i < path.length - 1; i+=2) {
            path[i] = writeMatrix(matrices[i/2].ravel());
            path[i+1] = titles.get(i/2);
        }
        String paths = StringUtils.join(path, ",");
//        String plotFile = "~/Documents/skymind/examples/dl4j-0.0.3.3-examples/src/main/resources/python/plot.py";
        String plotFile = new ClassPathResource("python/plot.py").getFile().getAbsolutePath();
        String cmd = "python " + plotFile + " multi "+ paths;

        log.info("Rendering Matrix histograms. Close window when ready ton continue or end program... ");
        Process is = Runtime.getRuntime().exec(cmd);

        log.info("Std out " + IOUtils.readLines(is.getInputStream()).toString());
        log.error("Std error " + IOUtils.readLines(is.getErrorStream()).toString());


        // Single layer just learns features and can't be supervised. Thus cannot be evaluated.


    }
}