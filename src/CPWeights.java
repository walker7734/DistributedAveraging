import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;


public class CPWeights{

    //volatile to allow for atomic updates
    volatile double w0;
    volatile Map<Integer, Double> wTokens;
    volatile double wPosition;
    volatile double wDepth;
    volatile double wAge;
    volatile double wGender;
    
    Map<Integer, Integer> accessTime; // keep track of the access timestamp of feature weights.
                                     // Using this to do delayed regularization.
    
    public CPWeights() {
        w0 = wAge = wGender = wDepth = wPosition = 0.0;
        wTokens = new HashMap<Integer, Double>();
        accessTime = new ConcurrentHashMap<Integer, Integer>();
    }

    @Override
    public String toString() {
        DecimalFormat myFormatter = new DecimalFormat("###.##");
        StringBuilder builder = new StringBuilder();
        builder.append("Intercept: " + myFormatter.format(w0) + "\n");
        builder.append("Depth: " + myFormatter.format(wDepth) + "\n");
        builder.append("Position: " + myFormatter.format(wPosition) + "\n");
        builder.append("Gender: " + myFormatter.format(wGender) + "\n");
        builder.append("Age: " + myFormatter.format(wAge) + "\n");
        return builder.toString();
    }

    /**
     * @return the l2 norm of this weight vector.
     */
    public double l2norm() {
        double l2 = w0 * w0 + wAge * wAge + wGender * wGender
                                + wDepth*wDepth + wPosition*wPosition;
        for (int token : wTokens.keySet()) {
            double weight = wTokens.get(token);
            l2 +=  weight * weight;
        }
        return Math.sqrt(l2);
    }

    /**
     * @return the l0 norm of this weight vector.
     */
    public int l0norm() {
        return 4 + wTokens.size();
    }
}
