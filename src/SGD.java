import java.util.ArrayList;

public class SGD {
    private CPDataSet data;
    private double lambda;
    private CPWeights weights;
    private double step;
    private int timestamp;
    private int dataIndex;

    public SGD(CPDataSet dataSet, double step, double lambda) {
        data = dataSet;
        this.lambda = lambda;
        this.step = step;
        weights = new CPWeights();
        timestamp = 0;
    }

    /**
     * Helper function to compute inner product w^Tx.
     *
     * @param instance
     * @return
     */
    public static double computeWeightFeatureProduct(CPWeights localWeights, CPDataInstance instance) {
        double innerProduct;
        innerProduct = localWeights.w0 + instance.depth * localWeights.wDepth +
                instance.position * localWeights.wPosition;
        innerProduct += instance.gender * localWeights.wGender;
        innerProduct += instance.age * localWeights.wAge;
        int[] tokens = instance.tokens;
        for (int i = 0; i < tokens.length; i++) {
            if (!localWeights.wTokens.containsKey(tokens[i])) {
                localWeights.wTokens.put(tokens[i], 0.0);
            }

            innerProduct += localWeights.wTokens.get(tokens[i]);
        }
        return innerProduct;
    }

    /**
     * Apply delayed regularization to the weights corresponding to the given tokens.
     * @param tokens
     */
    private void performDelayedRegularization(int[] tokens) {

        for (int i = 0; i < tokens.length; i++) {
            if ( weights.accessTime.containsKey(tokens[i])) {
                double weight = weights.wTokens.get(tokens[i]);
                int exponent = timestamp - weights.accessTime.get(tokens[i]) - 1;
                double regularizePow = Math.pow((1.0 - lambda*step), exponent);
                weights.wTokens.put(tokens[i], weight*regularizePow);
            }
        }
    }

    private void preformFullRegularization(int now) {

        for (int token : weights.wTokens.keySet()) {
            double weight = weights.wTokens.get(token);
            int exponent = now - weights.accessTime.get(token) - 1;
            double regularizePow = Math.pow((1.0 - lambda*step), exponent);
            weights.wTokens.put(token, weight*regularizePow);
        }
    }


    public void calculateWeight(CPDataInstance feature,
                                       double gradient) {

        weights.w0 = weights.w0 + step*gradient;
        weights.wAge = weights.wAge*(1 - lambda*step) + step*(feature.age*gradient);
        weights.wGender = weights.wGender*(1 - lambda*step) + step*(feature.gender*gradient);
        weights.wPosition = weights.wPosition*(1 - lambda*step) + step*(feature.position*gradient);
        weights.wDepth = weights.wDepth*(1 - lambda*step) + step*(feature.depth*gradient);

        //update tokens
        int[] tokens = feature.tokens;
        for (int i = 0; i < tokens.length; i++) {
            double tokenWeight = weights.wTokens.get(tokens[i]);
            weights.wTokens.put(tokens[i], tokenWeight*(1 - lambda*step) + (step*gradient));
            weights.accessTime.put(tokens[i], timestamp);
        }
    }

    public static double calculateProbability(double weightProduct) {
        return Math.exp(weightProduct) / (1.0 + Math.exp(weightProduct));
    }


    /**
     * Train the logistic regression model using the training data and the
     * hyperparameters. Return the weights, and record the cumulative loss.
     *
     * @return the weights for the model.
     */
    public void train() {
        CPDataInstance dataPoint = (CPDataInstance)data.getInstanceAt(dataIndex);

        //increment the current time
        timestamp++;
        dataIndex++;

        //perform delayed regularization
        //performDelayedRegularization(dataPoint.tokens);

        //calculate the probability
        double innerProduct = computeWeightFeatureProduct(weights, dataPoint);
        double prob = calculateProbability(innerProduct);

        //calculate the gradient
        int y = dataPoint.clicked;
        double gradient = (y - prob);

        //update weights
        calculateWeight(dataPoint, gradient);

    }

    private void reset() {
        timestamp = 0;
        weights = new CPWeights();
        dataIndex = 0;
    }

    public CPWeights run(int iterations) {
        reset();
        for (int i = 0; i < iterations; i++) {
            dataIndex = 0;
            for (int j = 0; j < data.getSize(); j++) {
                train();
            }
        }
        //preformFullRegularization(timestamp);
        return weights;
    }
}