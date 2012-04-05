package org.ikit.nlp;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Map.Entry;
import java.util.*;


/**
 * TF-IDF algorithm in Java. An explanation of this algorithm can be found at
 * 
 *      http://en.wikipedia.org/wiki/Tf*idf
 * 
 * @author Bodong Chen <bodong.chen@gmail.com>
 */
public class TF_IDF {
    
    private Set<String> stopwords;      // a common set of English stopwords

    private List<List<String>> docs;    // documents as bags of words, with stopwords removed
    private int numDocs;
    
    private ArrayList<String> terms;    // unique terms
    private int numTerms;
    
    private int[][] termFreq;           // tf matrix
    private double[][] termWeight;      // tf-idf matrix
    private int[] docFreq;              // terms' frequency in all documents
    
    /**
     * Constructor
     * @param documents documents represented as strings
     */
    public TF_IDF(String[] documents) {
        
        stopwords = this.loadStopWords("stoplist.txt");
        
        docs = this.parseDocuments(documents);
        numDocs = docs.size();
        
        terms = this.generateTerms(docs);
        numTerms = terms.size();
        
        docFreq = new int[numTerms];
        termFreq = new int[numTerms][numDocs];
        termWeight = new double[numTerms][numDocs];

        this.countTermOccurrence();
        this.generateTermWeight();
    }
    
    /**
     * Load stopwords from a file
     * @param filename
     * @return 
     */
    private Set<String> loadStopWords(String filename) {
        Set<String> stoplist = new HashSet<String>();

        try {
            InputStream in = this.getClass().getResourceAsStream(filename);
            BufferedReader br = new BufferedReader(new InputStreamReader(in));
            String line;
            while ((line = br.readLine()) != null) {
                stoplist.add(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return stoplist;
    }
    
    /**
     * Parse documents into bags of words
     * @param docs documents in strings
     * @return a list of documents represented by bags of words
     */
    private List<List<String>> parseDocuments(String[] docs) {
        List<List<String>> parsedDocs = new ArrayList<List<String>>();
        
        for(String doc: docs) {
            String[] words = doc.replaceAll("\\p{Punct}", "")
                    .toLowerCase().split("\\s");
            List<String> wordList = new ArrayList<String>();
            for(String word: words) {
                word = word.trim();
                if (word.length() > 0 && !stopwords.contains(word)) {
                    wordList.add(word);
                }
            }
            parsedDocs.add(wordList);
        }
        
        return parsedDocs;
    }
    
    /**
     * Generate terms from a list of documents
     * @param docs
     * @return 
     */
    private ArrayList<String> generateTerms(List<List<String>> docs) {
        ArrayList<String> uniqueTerms = new ArrayList();
        for(List<String> doc: docs) {
            for(String word: doc) {
                if (!uniqueTerms.contains(word)) {
                    uniqueTerms.add(word);
                }
            }
        }
        return uniqueTerms;
    }
    
    /**
     * Count term occurrence
     * and occurrence of each term in the whole corpus
     */
    private void countTermOccurrence() {
        for (int i = 0; i < docs.size(); i++) {
            List<String> doc = docs.get(i);
            HashMap<String, Integer> tfMap = this.countTermOccurrenceInOneDoc(doc);
            for(Entry<String, Integer> entry: tfMap.entrySet()) {
                String word = entry.getKey();
                int wordFreq = entry.getValue();
                int termIndex = terms.indexOf(word);
                
                termFreq[termIndex][i] = wordFreq;
                docFreq[termIndex]++;
            }
        }
    }
    
    /**
     * Count term frequency in a document
     * @param doc a document as a bag of words
     * @return a map of term occurrence; key - term; value - occurrence.
     */
    private HashMap<String, Integer> countTermOccurrenceInOneDoc(List<String> doc) {
        
        HashMap<String, Integer> tfMap = new HashMap<String, Integer>();
        
        for(String word: doc) {
            int count = 0;
            for(String str: doc) {
                if (str.equals(word)) {
                    count++;
                }
            }
            
            tfMap.put(word, count);
        }

        return tfMap;
    }
    
    /**
     * Calculate term weight based on tf*idf algorithm
     * There are different choices in calculating tf and idf.
     * So you may want to change it to fit your own needs.
     */
    private void generateTermWeight()
    {
        for (int i = 0; i < numTerms; i++) {
            for (int j = 0; j < numDocs; j++) {
                double tf = this.getTFMeasure(i, j);
                double idf = this.getIDFMeasure(i);
                termWeight[i][j] = tf * idf;
            }
        }
    }
    
    private double getTFMeasure(int term, int doc) {
        int freq = termFreq[term][doc];
        return Math.sqrt((double) freq);
    }

    private double getIDFMeasure(int term) {
        int df = docFreq[term];
        return 1.0d + Math.log( (double) (numDocs) / (1.0d + df) );
    }
    
    /**
     * Get similarity score between two documents
     * @param doc_i index of one document
     * @param doc_j index of another document
     * @return similarity score
     */
    public double getSimilarity(int doc_i, int doc_j) {
        double[] vector1 = this.getDocumentVector(doc_i);
        double[] vector2 = this.getDocumentVector(doc_j);
        return TF_IDF.computeCosineSimilarity(vector1, vector2);
    }
    
    /**
     * Compile a vector for a document
     * @param docIndex index of a document
     * @return the vector representation of the document
     */
    private double[] getDocumentVector(int docIndex) {
        double[] v = new double[numTerms];
        for (int i = 0; i < numTerms; i++) {
            v[i] = termWeight[i][docIndex];
        }
        return v;
    }

    /**
     * Calculate cosine similarity between two vectors
     * @param vector1 a vector
     * @param vector2 another vector
     * @return cosine similarity score
     */
    public static double computeCosineSimilarity(double[] vector1, double[] vector2)
    {
        if (vector1.length != vector2.length) {
            System.out.println("Different vector length.");
        }

        double denom = (vectorLength(vector1) * vectorLength(vector2));
        if (denom == 0.0d) {
            return 0.0d;
        } else {
            return (innerProduct(vector1, vector2) / denom);
        }
    }

    /**
     * Calculate inner product of two vectors
     * @param vector1 a vector
     * @param vector2 another vector
     * @return inner production of two vectors
     */
    public static double innerProduct(double[] vector1, double[] vector2)
    {
        double result = 0.0d;
        for (int i = 0; i < vector1.length; i++) {
            result += vector1[i] * vector2[i];
        }
        return result;
    }

    /**
     * Calculate vector length
     * @param vector a vector
     * @return vector length
     */
    public static double vectorLength(double[] vector)
    {
        double sum = 0.0d;
        for(double d: vector) {
            sum += d * d;
        }
        return Math.sqrt(sum);
    }
    
    
    /**
     * Testing
     * @param args 
     */
    public static void main(String[] args) {
        String[] docs = {"knowledge building needs innovative environments are better at helping their inhabitants explore the adjacent possible",
                        "As a basis for evaluating explanations, creative knowledge building weight of evidence is a poor substitute for the first two criteria listed above.",
                        "A public idea database makes every passing idea visible to everyone else in the organization and do creative work.",
                        "questioning and various disturbances initiate cycles of innovation and creative organization knowledge.",
                        "We need some way to ensure knowledge to spread among environments that any notes that are dropped are dropped."};
        
        TF_IDF tfIdf = new TF_IDF(docs);
        for(int i = 0; i < tfIdf.docs.size(); i++) {
            System.out.print(i+1 + "\t");
            for (int j = 0; j < tfIdf.docs.size(); j++) {
                System.out.print(tfIdf.getSimilarity(i, j) + "\t");
            }
            System.out.println();
        }
    }
}
