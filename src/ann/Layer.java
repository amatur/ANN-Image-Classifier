package ann;

import java.util.ArrayList;
import org.jblas.DoubleMatrix;

/**
 *
 * @author Hera
 */
public class Layer {
    
    Neuron[] neurons;
    DoubleMatrix[] weightsOfNeurons;
    
    Layer(int numNeurons, int numNeuronsInPrevLayer) {
        neurons = new Neuron[numNeurons];
        weightsOfNeurons = new DoubleMatrix[numNeurons];
        ArrayList<Double> list = new ArrayList<Double>();
        for (int i = 0; i <= numNeuronsInPrevLayer; i++)
            list.add(1.0);
        for (int i = 0; i < neurons.length; i++)
            weightsOfNeurons[i] = new DoubleMatrix(list);
        for (int i = 0; i < neurons.length; i++)
            neurons[i] = new Neuron (weightsOfNeurons[i]);
    }
    
    DoubleMatrix calculateYs(DoubleMatrix inputs) {
        ArrayList<Double> l = new ArrayList<Double>(inputs.elementsAsList());
        l.add(1.0);
        inputs = new DoubleMatrix (l);
        ArrayList<Double> list = new ArrayList<Double>();
        for (Neuron n : neurons) {
            Double d = n.calculateY(inputs);
            list.add(d);
        }
        return new DoubleMatrix(list);
    }
    
    DoubleMatrix calculateVs(DoubleMatrix inputs) {
        ArrayList<Double> l = new ArrayList<Double>(inputs.elementsAsList());
        l.add(1.0);
        inputs = new DoubleMatrix (l);
        ArrayList<Double> list = new ArrayList<Double>();
        for (Neuron n : neurons) {
            Double d = n.calculateV(inputs);
            list.add(d);
        }
        return new DoubleMatrix(list);
    }
    
    DoubleMatrix calculateYsFromVs(DoubleMatrix V) {
        ArrayList<Double> l = new ArrayList<Double>(V.elementsAsList());
        ArrayList<Double> list = new ArrayList<Double>();
        for (int i = 0; i < l.size(); i++) {
            list.add(neurons[i].calculateYFromV(l.get(i)));
        }
        return new DoubleMatrix(list);
    }
    
    DoubleMatrix calculateDeivativeYs(DoubleMatrix V) {
        ArrayList<Double> l = new ArrayList<Double>(V.elementsAsList());
        ArrayList<Double> list = new ArrayList<Double>();
        for (int i = 0; i < l.size(); i++) {
            list.add(neurons[i].calculateDerivativeYFromV(l.get(i)));
        }
        return new DoubleMatrix(list);
    }
    
    void print() {
        System.out.println("Layer with " + neurons.length + " neurons");
        for (Neuron n : neurons) {
            n.print();
        }
    }
    
}
