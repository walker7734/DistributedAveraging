import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Created with IntelliJ IDEA.
 * User: bdwalker
 * Date: 3/12/13
 * Time: 2:02 AM
 * To change this template use File | Settings | File Templates.
 */
public class DistributedAveraging {
    private static CPDataSet data;
    private int threads;
    private volatile static CPWeights weights;
    private volatile static int startCopyIndex;

    public DistributedAveraging(CPDataSet data) {
        DistributedAveraging.data = data;
        weights = new CPWeights();
        startCopyIndex = 0;
    }

    public ArrayList<Double> predict(CPDataSet testSet,CPWeights predictionWeights) {
        ArrayList<Double> prediction = new ArrayList<Double>();
        for (int i = 0; i < testSet.getSize(); i++) {
            CPDataInstance instance = (CPDataInstance)testSet.getInstanceAt(i);
            double weightProduct = SGD.computeWeightFeatureProduct(predictionWeights, instance);
            prediction.add(SGD.calculateProbability(weightProduct));
        }

        return prediction;
    }

    public synchronized void updateWeights(CPWeights localWeights) {
        DistributedAveraging.weights.w0 += localWeights.w0/(threads);
        DistributedAveraging.weights.wAge += localWeights.wAge/(threads);
        DistributedAveraging.weights.wDepth += localWeights.wDepth/(threads);
        DistributedAveraging.weights.wPosition += localWeights.wPosition/(threads);
        DistributedAveraging.weights.wGender += localWeights.wGender/(threads);

        for (int token : localWeights.wTokens.keySet()) {
            if (!DistributedAveraging.weights.wTokens.containsKey(token)) {
                DistributedAveraging.weights.wTokens.put(token, 0.0);
            }
            double tokenWeight = DistributedAveraging.weights.wTokens.get(token);
            DistributedAveraging.weights.wTokens.put(token,
                    tokenWeight + localWeights.wTokens.get(token)/(threads));
        }
    }

    public CPWeights run(final double step, final double lambda, final int fullIterations, int threads)
            throws InterruptedException {
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        this.threads = threads;

        //divy up data
        final int dataSize = data.getSize()/threads;

        //initialize each thread, should be equal to max number of cores for max
        //efficiency
        for (int thread = 0; thread < threads; thread++) {
            final int startIndex = thread;
            pool.submit(new Runnable() {
                @Override
                public void run() {
                    long threadId = Thread.currentThread().getId();
                    System.err.println("Spawning thread " + threadId);
                    CPDataSet localSet = new CPDataSet(dataSize);
                    localSet.copy(data, dataSize * startIndex, dataSize);
                    SGD sgd = new SGD(localSet , step, lambda);

                    System.err.println("Thread " + threadId + " starting SGD with data size " + dataSize);
                    CPWeights localWeights = sgd.run(fullIterations);

                    System.err.println("Thread " + threadId + " updating weights");
                    System.err.println("Thread " + threadId + " weights:");
                    System.err.println(localWeights);
                    updateWeights(localWeights);
                }
            });
        }

        pool.shutdown();

        pool.awaitTermination(120, TimeUnit.HOURS);
        return weights;
    }
}
