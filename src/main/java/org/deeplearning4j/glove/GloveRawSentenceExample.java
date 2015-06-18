package org.deeplearning4j.glove;

import org.deeplearning4j.bagofwords.vectorizer.TextVectorizer;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.models.glove.CoOccurrences;
import org.deeplearning4j.models.glove.Glove;
import org.deeplearning4j.models.glove.GloveWeightLookupTable;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.plot.Tsne;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.util.Collection;

/**
 * Global Vectors for Word Representation (http://nlp.stanford.edu/projects/glove/)
 *
 * Created by agibsonccc on 10/9/14.
 */
public class GloveRawSentenceExample {

    private static Logger log = LoggerFactory.getLogger(GloveRawSentenceExample.class);

    public static void main(String[] args) throws Exception {
        // Customizing params
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        int batchSize = 1000;
        int iterations = 30;
        int layerSize = 300;

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
        InMemoryLookupCache cache = new InMemoryLookupCache();
        TokenizerFactory tokenizer = new DefaultTokenizerFactory();
        tokenizer.setTokenPreProcessor(preProcessor);
        TextVectorizer  tfidf = new TfidfVectorizer.Builder()
                .cache(cache)
                .iterate(iter)
                .minWords(1)
                .tokenize(tokenizer)
                .build();
        tfidf.fit();

        log.info("Build model....");
        CoOccurrences coOccur = new CoOccurrences.Builder()
                .cache(cache)
                .iterate(iter)
                .tokenizer(tokenizer)
                .build();
        coOccur.fit();

        GloveWeightLookupTable table = new GloveWeightLookupTable.Builder()
                .cache(cache)
                .lr(0.005)
                .build();

        Glove vec = new Glove.Builder()
                .learningRate(0.005)
                .batchSize(batchSize)
                .cache(cache)
                .coOccurrences(coOccur)
                .iterations(iterations)
                .vectorizer(tfidf)
                .weights(table)
                .layerSize(layerSize)
                .iterate(iter)
                .tokenizer(tokenizer)
                .minWordFrequency(1)
                .symmetric(true)
                .windowSize(15)
                .build();

        log.info("Train model....");
        vec.fit();

        log.info("Evaluate model....");
        double sim = vec.similarity("people", "money");
        log.info("Similarity between people and money: " + sim);
        Collection<String> similar = vec.wordsNearest("day",20);
        log.info("Similar words to 'day' : " + similar);

        log.info("Store TSNE....");
        Tsne tsne = new Tsne.Builder().setMaxIter(200)
                .learningRate(500).useAdaGrad(false)
                .normalize(false).usePca(false).build();
        vec.lookupTable().plotVocab(tsne);

        log.info("****************Example finished********************");


    }

}
