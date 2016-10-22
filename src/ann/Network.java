/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ann;

import static ann.ANN.AbaloneTrain;
import java.util.ArrayList;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

/**
 *
 * @author Tasnim
 */
public class Network {

    int numLayers;
    Layer[] layers;
    int minibatchsize;
    ArrayList<Example> list;

    public Network(ArrayList<Example> list, int[] layers, int minibatchsize) {
        this.layers = new Layer[numLayers];
        this.minibatchsize = minibatchsize;
        this.numLayers = layers.length;
        this.list = list;
        this.layers = new Layer[numLayers];

        /**
         * *** input layer (0), hidden layers, output layer (numLayers-1)***
         */
        for (int i = 0; i < numLayers - 1; i++) {
            // input layer
            if (i == 0) {
                System.out.println("Initializing input layer with size " + layers[i]);

                //add bias
                this.layers[0] = new Layer(layers[0] + 1, layers[1], true, false, minibatchsize);
            } // output layer
            else if (i == numLayers - 1) {
                System.out.println("Initializing output layer with size " + layers[i]);
                //no bias is added
                this.layers[i] = new Layer(layers[i], null, false, true, minibatchsize);
                // you have to do softmax in this layer
            } else {
                System.out.printf("Initializing hidden layer %d layer with size %d \n", i, layers[i]);

                //add bias
                this.layers[i] = new Layer(layers[i] + 1, layers[i + 1], false, false, minibatchsize);
            }
        }

        System.out.printf("Network creation done!\n");

    }

    RealMatrix forward_propagate() {
        //number of examples, n
        int n = list.size();

        //number of features + 1 for bias, m
        int m = AbaloneLoader.NUM_FEATURES + 1;

        double[][] nmMat = new double[n][m];
        for (int i = 0; i < n; i++) {
            Example ex = list.get(i);
            for (int j = 0; j < m - 1; j++) {
                nmMat[i][j] = ex.features.get(j);
            }
            //add bias
            for (int j = m - 1; j < m; j++) {
                nmMat[i][j] = 1;
            }
        }
        //X => n * m matrix -> example vs. features
        RealMatrix X = MatrixUtils.createRealMatrix(nmMat);
        this.layers[0].Z = X;   //input layer

        for (int i = 0; i < numLayers - 1; i++) {
            layers[i + 1].S = layers[i].forwardPropagate();
        }
        return layers[numLayers - 1].forwardPropagate();

    }

    public void doit() {

//        
//        
//        
//        //number of examples, n
//        int n = list.size();
//        
//        //number of feature, m
//        int m = AbaloneLoader.NUM_FEATURES ;
//        
//        double[][] nmMat= new double[n][m];
//        for (int i = 0; i < n; i++) {
//            Example ex  = list.get(i);
//            for (int j = 0; j < m; j++) {                
//                nmMat[i][j] = ex.features.get(j);
//            }
//        }
//        //X => n * m matrix -> example vs. features
//        RealMatrix X = MatrixUtils.createRealMatrix(nmMat);
//        
//        //System.out.println("FFF" + X.getRowDimension());
//        
//        //W => m * a matrix -> num of features vs. weights
//        double[] layer1Weights = {5, 2, 3};
//        double[][] Wmat = new double[m][layer1Weights.length];
//        for (int i = 0; i < m; i++) {
//            for (int j = 0; j < layer1Weights.length; j++) {                
//                Wmat[i][j] = layer1Weights[j];
//            }
//        }
//        RealMatrix W = MatrixUtils.createRealMatrix(Wmat);
//        
//        RealMatrix S = X.multiply(W);
//        RealMatrix Z = applyActivation(S, false);
//        
//        
//        System.out.println(S);        System.out.println(Z);
//
//        
//        System.out.println(S.getColumnDimension());
//         System.out.println(S.getRowDimension());
    }

}
