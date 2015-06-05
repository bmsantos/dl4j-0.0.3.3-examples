package org.deeplearning4j.recursive;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.models.rntn.RNTN;
import org.deeplearning4j.models.rntn.RNTNEval;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.layers.feedforward.autoencoder.recursive.Tree;
import org.deeplearning4j.text.corpora.treeparser.TreeVectorizer;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author Adam Gibson
 */
public class RNTNTweetsExample {

    private static final Logger log = LoggerFactory.getLogger(RNTNTweetsExample2.class);

    public static void main(String[] args) throws Exception {

        log.info("Load data....");

        List<String> lines = FileUtils.readLines(new ClassPathResource("sentiment-tweets-small.csv").getFile());
        List<String> sentences = new ArrayList<>();
        List<String> labels = new ArrayList<>();
        int count = 0;
        for(String s : lines) {
            if (count > 0) {
                labels.add(s.split(",")[1]);
                sentences.add(s.split(",")[3]);
            }
            count++;
        }

        log.info("Vectorize data....");
        SentenceIterator iter = new CollectionSentenceIterator(sentences);
        Word2Vec vec = new Word2Vec.Builder()
                .batchSize(1000)
                .sampling(1e-5)
                .minWordFrequency(5)
                .useAdaGrad(false)
                .layerSize(300)
                .iterations(3)
                .learningRate(0.025)
                .minLearningRate(1e-2)
                .negativeSample(10)
                .iterate(iter)
                .build();
        vec.fit();
        iter.reset();

        log.info("Build model....");
        TreeVectorizer trees = new TreeVectorizer();
        // TODO verfiy rng works when fixed
        RNTN rntn = new RNTN.Builder().setActivationFunction("tanh")
                .setRng(new DefaultRandom(3))
                .setAdagradResetFrequency(1)
                .setCombineClassification(true)
                .setFeatureVectors(vec)
                .setRandomFeatureVectors(false)
                .setUseTensors(false)
                .build();
        count = 0;

        log.info("Train model....");
        while(iter.hasNext()) {
            String next = iter.nextSentence();
            List<Tree> treeList = trees.getTreesWithLabels(next, Arrays.asList(labels.get(count++)));
            rntn.fit(treeList);
        }

        log.info("Evaluate model....");
        count = 0;
        iter.reset();
        RNTNEval eval = new RNTNEval();
        while(iter.hasNext()) {
            String next = iter.nextSentence();
            List<Tree> treeList = trees.getTreesWithLabels(next, Arrays.asList(labels.get(count++)));
            eval.eval(rntn,treeList);
        }

        log.info(eval.stats());

        rntn.shutdown();

        log.info("****************Example finished********************");
    }

}
