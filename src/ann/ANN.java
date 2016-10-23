package ann;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;


/**
 *
 * @author Hera
 */
public class ANN {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        //creating data
        ArrayList<Example> list = MNISTDatasetTrain();
        //ArrayList<Example> list = AbaloneTrain();
        
        int[] layers = {MNISTLoader.NUM_FEATURES, 90, 80, MNISTLoader.NUM_CLASSES};
        //int[] layers = {AbaloneLoader.NUM_FEATURES, 90, 80, AbaloneLoader.NUM_CLASSES};
        int minibatchsize = 100;
        Network n = new Network(list, layers, minibatchsize);
        n.evaluate();
        //n.forwardPropagate(list);
        
        
        //Abalone
        //int[] layers = {AbaloneLoader.NUM_FEATURES, 10, 19, 28};
        
        
        //RealMatrix v = n.forwardPropagate();
       
        
        //System.out.println(v + " " + v.getColumnDimension() + " " + v.getRowDimension());
        
        //RealMatrix labelMatrix = MatrixUtils.createRealMatrix(list.size(), AbaloneLoader.NUM_RINGS);
        
        //for (int i = 0; i < list.size(); i++) {
        //    labelMatrix.setRow(i, list.get(i).outputs.toArray());
        //}
       // System.out.println(labelMatrix);
      
        //n.backPropagate(labelMatrix, labelMatrix);
        
        
        /*
         Scanner in = new Scanner(System.in);
         int[] nums = {5, 10};
        
         //Network net = new Network (MNISTLoader.NUM_CLASSES, nums, MNISTLoader.NUM_FEATURES);
         Network net = new Network (AbaloneLoader.NUM_RINGS, nums, AbaloneLoader.NUM_FEATURES);
         //Network net = new Network (2, nums, 2);
        
         ArrayList list = AbaloneTrain(400);
         //ArrayList list = MNISTDatasetTest(50);
         //ArrayList list = simpleDatasetGenerator();
        
         net.train(list , 0.4);
        
        
         System.out.println("Done training, start checking");
         //ArrayList<Example> al = simpleDatasetGenerator();
         ArrayList<Example> al = list;
         for (Example e : al) {
         DoubleMatrix result =  net.runInput(e.features);
         System.out.println("Result: " + result + " where actually " + e.outputs);
         System.out.println("Result: " + Example.getLabel(result) + " where actually " + Example.getLabel(e.outputs));
         }
         */
    }

//    public double update_weights(double eta, int num_layers){
//        for (int i = 0; i < num_layers-1; i++) {
//            W_grad = -eta*(layers[i+1].D.dot(layers[i].Z)).T;
//            layers[i].W += W_grad;
//        }
//    }
    
    
    
    
    static void splitTrainValidation(ArrayList<Example> list, double percent){
        percent = 0.7;
    }

    static void runTest(RealMatrix Y, ArrayList<Example> list){
        
    }
    
    static ArrayList<Example> MNISTDatasetTest(int setSize) {
        MNISTLoader loader = new MNISTLoader(MNISTLoader.TEST_LABEL, MNISTLoader.TEST_IMAGE);
       // ArrayList<Example> al = loader.getCompleteSubset(setSize);
         ArrayList<Example> al = loader.getExampleList();
        //System.out.println(al);
        return al;
    }

    static ArrayList<Example> MNISTDatasetTrain() {
        MNISTLoader loader = new MNISTLoader(MNISTLoader.TRAIN_LABEL, MNISTLoader.TRAIN_IMAGE);
        //ArrayList<Example> al = loader.getCompleteSubset(setSize);
         //ArrayList<Example> al = loader.getRandomSubset(1000);
         ArrayList<Example> al = loader.getExampleList();
        //System.out.println(al);
        return al;
    }

    static ArrayList<Example> AbaloneTrain() {
        AbaloneLoader loader = new AbaloneLoader(AbaloneLoader.ABALONE_TRAIN);
        ArrayList<Example> al = loader.getRandomSubset(1000);
        //System.out.println(al);
        return al;
    }

    static ArrayList<Example> simpleDatasetGenerator() {
        ArrayList<Example> list = new ArrayList<Example>();
        for (int i = 0; i < 50; i++) {
            double x = 2 * (Math.random() - 1);
            double y = 2 * (Math.random() - 1);
            double[] features = {x, y};
            double[] classif = {1.0, 0.0};
            list.add(new Example(features, classif));
        }
        for (int i = 0; i < 50; i++) {
            double x = 4 * (Math.random() - 1);
            double y = 4 * (Math.random() - 1);
            if (x * x < 1.3 || y * y < 1.3) {
                i--;
                continue;
            }
            double[] features = {x, y};
            double[] classif = {0.0, 1.0};
            list.add(new Example(features, classif));
        }
        for (Example e : list) {
            e.print();
        }

        return list;
    }

}
