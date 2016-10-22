package ann;

import java.util.ArrayList;
import org.jblas.DoubleMatrix;

/**
 *
 * @author Hera
 */
public class Example {
    
    DoubleMatrix features;
    DoubleMatrix outputs;

    public Example(DoubleMatrix features, DoubleMatrix outputs) {
        this.features = features;
        this.outputs = outputs;
    }
    
    public Example(double[] features, double[] outputs) {
        this.features = new DoubleMatrix (features);
        this.outputs = new DoubleMatrix (outputs);
    }
    
    public Example(double[] features, int label, int numLabels ) {
        this.features = new DoubleMatrix (features);        
        double arr[] = new double[numLabels];
        arr[label] = 1.0;        
        this.outputs = new DoubleMatrix (arr);
    }
    
    
    
     public int getLabel() {
        double max = Double.NEGATIVE_INFINITY;
        int label = -1;
        ArrayList<Double> al = new ArrayList<>(outputs.elementsAsList()) ;
        for (int i = 0; i < outputs.length; i++) {
            
            if(al.get(i) >= max){
                max = al.get(i);
                label = i;
            }
        }
        return label;
    }
   
    public static int getLabel(DoubleMatrix outputs) {
        double max = Double.NEGATIVE_INFINITY;
        int label = -1;
        ArrayList<Double> al = new ArrayList<>(outputs.elementsAsList()) ;
        for (int i = 0; i < outputs.length; i++) {
            
            if(al.get(i) >= max){
                max = al.get(i);
                label = i;
            }
        }
        return label;
    }
    
    void print() {
        System.out.println(features + " mapped to " + outputs);
    }
    
    @Override
    public String toString() {
        String str = "";
        ArrayList<Double> al = new ArrayList<>(features.elementsAsList()) ;
        for (Double d : al) {
            str += d + ", ";
        }
        return "Example{" + "featureVector=" + str + ", class=" + getLabel(this.outputs) + '}' + "\n";
    }
    
}
