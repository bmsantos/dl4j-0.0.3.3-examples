package org.deeplearning4j.word2vec;

import org.deeplearning4j.clustering.sptree.DataPoint;
import org.deeplearning4j.clustering.vptree.VPTree;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.SerializationUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Created by agibsonccc on 10/9/14.
 */
public class Word2VecRawTextExample {

    private static Logger log = LoggerFactory.getLogger(Word2VecRawTextExample.class);

    public static void main(String[] args) throws Exception {
        // Customizing params
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        int layerSize = 100;
        Nd4j.getRandom().setSeed(123);
        log.info("Load data....");
        ClassPathResource resource = new ClassPathResource("raw_sentences.txt");
        SentenceIterator iter = new LineSentenceIterator(resource.getFile());
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });

        log.info("Tokenize data....");
        final EndingPreProcessor preProcessor = new EndingPreProcessor();
        TokenizerFactory tokenizer = new DefaultTokenizerFactory();
        tokenizer.setTokenPreProcessor(new TokenPreProcess() {
            @Override
            public String preProcess(String token) {
                token = token.toLowerCase();
                String base = preProcessor.preProcess(token);
                base = base.replaceAll("\\d", "d");
                if (base.endsWith("ly") || base.endsWith("ing"))
                    System.out.println();
                return base;
            }
        });

        log.info("Build model....");
        Word2Vec vec = new Word2Vec.Builder()
                .batchSize(10000)
                .useAdaGrad(false)
                .layerSize(layerSize)
                .iterations(3).minWordFrequency(5)
                .learningRate(0.025)
                .minLearningRate(1e-2)
                .iterate(iter)
                .tokenizerFactory(tokenizer)
                .build();
        vec.fit();

        log.info("Evaluate model....");
        double sim = vec.similarity("people", "money");
        log.info("Similarity between people and money: " + sim);
        Collection<String> similar = vec.wordsNearest("day", 20);
        log.info("Similar words to 'day' : " + similar);
        INDArray dayVector = vec.getWordVectorMatrix("day");
        InMemoryLookupTable table = (InMemoryLookupTable) vec.lookupTable();

        VPTree tree = new VPTree(table.getSyn0(),"dot",true);
        List<DataPoint> results = new ArrayList<>();
        tree.search(new DataPoint(vec.vocab().indexOf("day"), dayVector, "dot", true), 20, results, new ArrayList<Double>());
        StringBuffer sb = new StringBuffer();
        for(DataPoint p :results)
            sb.append(vec.vocab().wordAtIndex(p.getIndex()) + ",");
        System.out.println(dayVector);
        log.info("Plot TSNE....");
        BarnesHutTsne tsne = new BarnesHutTsne.Builder()
                .setMaxIter(1000)
                .stopLyingIteration(250)
                .learningRate(500)
                .useAdaGrad(false)
                .theta(0.5)
                .setMomentum(0.5)
                .normalize(true)
                .usePca(false)
                .build();
        vec.lookupTable().plotVocab(tsne);

        log.info("Save vectors....");
        SerializationUtils.saveObject(vec, new File("vec.ser"));
        WordVectorSerializer.writeWordVectors(vec, "words.txt");

        log.info("****************Example finished********************");

    }

}
