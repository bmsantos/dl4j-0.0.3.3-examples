package org.deeplearning4j.tsne;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by agibsonccc on 10/9/14.
 *
 * Barnes-Hut better for large real-world datasets
 */
public class TSNEBarnesHutExample {

    private static Logger log = LoggerFactory.getLogger(TSNEBarnesHutExample.class);

    public static void main(String[] args) throws Exception {
        int iterations = 1000;
        List<String> cacheList = new ArrayList<>();

        log.info("Load & vectorize data....");
        File wordFile = new ClassPathResource("words.txt").getFile();
        Pair<InMemoryLookupTable,VocabCache> pair = WordVectorSerializer.loadTxt(wordFile);
        VocabCache vocabCache = pair.getSecond();
        INDArray weights = pair.getFirst().getSyn0();

        log.info("Build model....");
        BarnesHutTsne tsne = new BarnesHutTsne.Builder()
                .setMaxIter(iterations)
                .normalize(true)
                .stopLyingIteration(250)
                .learningRate(500)
                .theta(0.5)
                .setMomentum(0.5)
                .useAdaGrad(false)
                .usePca(false)
                .build();

        for(int i = 0; i < vocabCache.numWords(); i++)
            cacheList.add(vocabCache.wordAtIndex(i));

        log.info("Store Vocab TSNE....");
        tsne.plot(weights, 2, cacheList, "target/archive-tmp/tsne-barneshut-coords.csv");

    }

}
