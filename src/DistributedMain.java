import com.sun.xml.internal.ws.wsdl.writer.WSDLGenerator;

import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;

/**
 * Created with IntelliJ IDEA.
 * User: bdwalker
 * Date: 3/12/13
 * Time: 2:43 AM
 * To change this template use File | Settings | File Templates.
 */
public class DistributedMain {
    public static final int iterations = 5;

    public static void main(String[] args) throws IOException, InterruptedException {
        CPDataSet ds = new CPDataSet("data/train.txt", true);
        CPDataSet testData = new CPDataSet("data/test.txt", false);
        ArrayList<Double> ctr = parseCTR("data/test_label.txt");
        System.err.println("Finished loading data...");
        double averageError = 0;
        double averageRuntime = 0;
        PrintStream output = new PrintStream(new File("results.csv"));

        for (int cores = 1; cores <= 6; cores++) {
            averageError = 0;
            averageRuntime = 0;
            for (int i = 0; i < iterations; i++) {
                DistributedAveraging distAvg = new DistributedAveraging(ds);
                long start = System.currentTimeMillis();
                CPWeights daWeights = distAvg.run(.001, 0, 4, cores);
                long end = System.currentTimeMillis();
                System.err.println("Final Weights:");
                System.err.println(daWeights);
                ArrayList<Double> daPrediction = distAvg.predict(testData, daWeights);
                double daRMSE = ReportResults.getRMSE(daPrediction, ctr);
                averageError += daRMSE/iterations;
                long runtime = end - start;
                averageRuntime += runtime/iterations;
                System.out.println("RMSE: " + daRMSE);
                System.out.println("Runtime: " + runtime);
            }

            System.out.println("Average RMSE: " + averageError);
            System.out.println("Average runtime: " + averageRuntime);
            output.println(cores + "," + averageError + "," + averageRuntime);
        }
    }

    public static ArrayList<Double> parseCTR(String path) {
        try {
            Scanner input = new Scanner(new BufferedReader(new FileReader(path)));
            ArrayList<Double> ctr = new ArrayList<Double>();
            while (input.hasNextDouble()) {
                ctr.add(input.nextDouble());
            }
            return ctr;
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static void printResults(CPWeights weights) {
        System.out.println("Weight Results");
        System.out.println("Intercept: \t" + weights.w0);
        System.out.println("Age: \t\t" + weights.wAge);
        System.out.println("Position: \t\t" + weights.wPosition);
        System.out.println("Depth: \t\t" + weights.wDepth);
        System.out.println("Gender: \t\t" + weights.wGender);
    }
}
