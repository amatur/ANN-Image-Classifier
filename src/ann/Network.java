package ann;

import java.util.ArrayList;
import org.jblas.DoubleMatrix;

/**
 *
 * @author Hera
 */
public class Network {
    
    Layer[] layers;
    
    Network(int numClasses, int[] numNeuronsInHiddenLayer, int numFeaturesInInputLayer) {
        layers = new Layer[numNeuronsInHiddenLayer.length + 1];
        for (int i = 0; i < layers.length; i++) {
            if (i == 0) {
                layers[i] = new Layer (numNeuronsInHiddenLayer[i], numFeaturesInInputLayer);
            } else if (i < layers.length - 1) {
                layers[i] = new Layer (numNeuronsInHiddenLayer[i], numNeuronsInHiddenLayer[i - 1]);
            } else {
                layers[i] = new Layer (numClasses, numNeuronsInHiddenLayer[i - 1]);
            }
        }
    }
    
    DoubleMatrix runInput(DoubleMatrix features) {
        DoubleMatrix returnMatrix = features;
        for (Layer l : layers) {
            returnMatrix = l.calculateYsFromVs(l.calculateVs(returnMatrix));
        }
        return returnMatrix;
    }
    
    void train(ArrayList<Example> trainList, double learningRate) {
        int n = layers.length;
        for (int iterCount = 0; iterCount < 500000; iterCount++) {
            DoubleMatrix[][] weightUpdateAmount = new DoubleMatrix[n][];
            for (int i = 0; i < weightUpdateAmount.length; i++) {
                weightUpdateAmount[i] = new DoubleMatrix[layers[i].neurons.length];
            }
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < layers[i].neurons.length ; j++) {
                    weightUpdateAmount[i][j] = DoubleMatrix.zeros(layers[i].neurons[j].weights.data.length);
                }
            }
            for (Example e : trainList) {
                DoubleMatrix[] deltas = new DoubleMatrix[n];
                DoubleMatrix[] Vs = new DoubleMatrix[n];
                DoubleMatrix[] Ys = new DoubleMatrix[n];
                DoubleMatrix[] derivativesYs = new DoubleMatrix[n];
                for (int i = 0; i < Vs.length; i++) {                    
                    if (i == 0)
                        Vs[i] = layers[i].calculateVs(e.features);
                    else
                        Vs[i] = layers[i].calculateVs(Ys[i - 1]);
                    Ys[i] = layers[i].calculateYsFromVs(Vs[i]);
                    derivativesYs[i] = layers[i].calculateDeivativeYs(Vs[i]);
                }
                deltas[n - 1] = Ys[n - 1].sub(e.outputs);
                for (int i = 0; i < deltas[n - 1].data.length; i++) {
                    deltas[n - 1].data[i] *= derivativesYs[n - 1].data[i];
                }
                for (int r = n - 1; r > 0; r--) {
                    DoubleMatrix m = layers[r].weightsOfNeurons[0];
                    for (int i = 1; i < layers[r].weightsOfNeurons.length; i++) {
                        m = DoubleMatrix.concatHorizontally(m, layers[r].weightsOfNeurons[i]);
                    }
                    ArrayList<DoubleMatrix> dmArray = new ArrayList(m.rowsAsList());
                    m = dmArray.get(0);
                    for (int i = 1; i < dmArray.size()-1; i++) {
                        m = DoubleMatrix.concatVertically(m, dmArray.get(i));
                    }
                    // TODO ... debug point
                    try {
                        deltas[r - 1] = (deltas[r].transpose().mmul(m.transpose())).transpose();
                    } catch (Exception ex) {
                        ex.printStackTrace();
                        System.out.println(r + " " + n + "\n" + deltas[r].transpose() + "\n" + m.transpose());
                    }
                    for (int i = 0; i < deltas[r - 1].data.length; i++) {
                        try {
                            deltas[r - 1].data[i] *= derivativesYs[r - 1].data[i];
                        } catch (Exception ex) {
                            System.out.println(deltas[r - 1] + "\n" + derivativesYs[r - 1]);
                            ex.printStackTrace();
                            System.exit(-1);
                        }
                    }
                }
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < layers[i].neurons.length; j++) {
                        ArrayList l;
                        if ( i == 0 ) {
                            l = new ArrayList(e.features.elementsAsList());
                        } else {
                            l = new ArrayList(Ys[i - 1].elementsAsList());
                        }
                        l.add(1.0);
                        DoubleMatrix f = new DoubleMatrix (l);
                        weightUpdateAmount[i][j] = weightUpdateAmount[i][j].add( f.mmul(deltas[i].data[j]) );
                    }
                }
            }
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < layers[i].neurons.length; j++) {
                    layers[i].neurons[j].weights = layers[i].neurons[j].weights.add( weightUpdateAmount[i][j].mmul(-learningRate) );
                }
            }

        }
    }
    
    void print() {
        for (Layer l : layers) {
            l.print();
        }
    }
    
}
