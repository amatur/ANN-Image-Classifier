package ann;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;


/**
 *
 * @author Hera
 */
public class ANN {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        Scanner in = new Scanner(System.in);
        int[] nums = {10, 10, 10};
        //Network net = new Network (2, nums, 2);
        Network net = new Network (MNISTLoader.NUM_CLASSES, nums, MNISTLoader.NUM_FEATURES);
        
        ArrayList list = MNISTDatasetGenerator();
        //net.train(simpleDatasetGenerator(), 0.1);
        net.train(list , 1.0);
        
        
        System.out.println("Done training, start checking");
        //ArrayList<Example> al = simpleDatasetGenerator();
        ArrayList<Example> al = list;
        for (Example e : al) {
            System.out.println("Result: " + net.runInput(e.features) + " where actually " + e.outputs);
        }
        
    }
    
    
    static ArrayList<Example> MNISTDatasetGenerator() {
        MNISTLoader loader = new MNISTLoader(MNISTLoader.TEST_LABEL, MNISTLoader.TEST_IMAGE);
        ArrayList<Example> al = loader.getRandomSubset(2);
        System.out.println(al); 
        return al;
    }
    
    static ArrayList<Example> simpleDatasetGenerator() {
        ArrayList<Example> list = new ArrayList<Example>();
        for (int i = 0; i < 50; i++) {
            double x = 2*(Math.random() - 1);
            double y = 2*(Math.random() - 1);
            double[] features = {x, y};
            double[] classif = {1.0, 0.0};
            list.add(new Example(features, classif));
        }
        for (int i = 0; i < 50; i++) {
            double x = 4*(Math.random() - 1);
            double y = 4*(Math.random() - 1);
            if (x*x < 1.3 || y*y < 1.3) {
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
