package ann;

import java.util.ArrayList;
import org.jblas.DoubleMatrix;

/**
 *
 * @author Hera
 */
public class Neuron {
    
    DoubleMatrix weights;
    
    Neuron(int numInputs) {
        ArrayList<Double> list = new ArrayList<Double>();
        for (int i = 0; i <= numInputs; i++)
            list.add(1.0);
        weights = new DoubleMatrix(list);
    }
    
    Neuron(DoubleMatrix weights) {
        this.weights = weights;
    }
    
    double calculateV(DoubleMatrix inputs) {
        assert inputs.length == weights.length;
        this.weights.assertSameSize(inputs);
        return this.weights.transpose().mmul(inputs).data[0];
    }
    
    double activation(double x, double a) {
        return 1.0 / (1 + Math.exp(-a * x));
    }
    
    double activation(double x) {
        return activation(x, 1);
    }
    
    double calculateYFromV(double v) {
        return activation(v);
    }
    
    double calculateY(DoubleMatrix inputs) {
        assert inputs.length == weights.length;
        return activation(calculateV(inputs));
    }
    
    double calculateDerivativeYFromV(double v) {
        double fX = activation(v);
        return fX * (1 - fX);
    }
    
    void print() {
        System.out.println("Neuron: " + weights.toString());
    }
    
}
