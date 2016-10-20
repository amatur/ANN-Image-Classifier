package ann;

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
    
    void print() {
        System.out.println(features + " mapped to " + outputs);
    }
    
}
