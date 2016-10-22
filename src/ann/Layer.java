/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ann;

import java.util.Random;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.*;
/**
 *
 * @author Tasnim
 */
public class Layer {
    RealMatrix Z, W, D, S, Fp;
    // W is the outgoing weight matrix for this layer
    // Z is the matrix that holds output values  
    // S is the matrix that holds the inputs to this layer
    // D is the matrix that holds the deltas for this layer
    // Fp is the matrix that holds the derivatives of the activation function
    boolean input;
    boolean output; 
    Integer numNodesThisLayer;
    Integer numNodesNextLayer;
    int minibatchSize;

    
    public Layer(Integer numNodesThisLayer, Integer numNodesNextLayer, boolean input, boolean output, int minibatchSize) {
        this.input = input;
        this.output = output;
        this.numNodesThisLayer = numNodesThisLayer;
        this.numNodesNextLayer = numNodesNextLayer;
        this.minibatchSize = minibatchSize;
        Random random = new Random(0);
        if (!input){
            S = MatrixUtils.createRealMatrix(minibatchSize, numNodesThisLayer);
            D = MatrixUtils.createRealMatrix(minibatchSize, numNodesThisLayer);
        }
        if (!output){
            W = MatrixUtils.createRealMatrix(numNodesThisLayer, numNodesNextLayer);            
            
            for (int i = 0; i < W.getRowDimension(); i++) {
                for (int j = 0; j < W.getColumnDimension(); j++) {
                    W.setEntry(i, j, getGaussian(random, 0, 1, 0.0001));
                }
            }
            System.out.printf("weight from layer %d to %d is: \n", numNodesThisLayer, numNodesNextLayer);
            System.out.println(W);
            //System.out.println(getGaussian(random, 0, 1, 0.0001));
            
        }
        
        if (input==false && output==false){
            Fp = MatrixUtils.createRealMatrix(numNodesThisLayer, minibatchSize);   
        }
           
    }
    
     private double getGaussian(Random random, double aMean, double aVariance, double aScale) {
        return (aMean + random.nextGaussian() * aVariance)*aScale;
     }
    
    
    RealMatrix forwardPropagate(){
        if(input == true){
            return Z.multiply(W);
        }
        Z = applyActivation(S, false);
        if(output == true){
            return Z;
        }else{
            //hidden layer
            
            //add bias, make a new column at the end, Z = numexamples * numnodesThisLayer
            double[][] newZ = new double[Z.getRowDimension()][Z.getColumnDimension()+1];
            for (int i = 0; i < Z.getRowDimension(); i++) {
                newZ[i][Z.getColumnDimension()] = 1;
            }
            Z = MatrixUtils.createRealMatrix(newZ);
            Fp = applyActivation(S, true).transpose();
            return Z.multiply(W);
        }
    }
    
    
    public RealMatrix applyActivation(RealMatrix X, boolean deriv){
        double[][] mat = new double[X.getRowDimension()][X.getColumnDimension()];
        for (int i = 0; i < X.getRowDimension(); i++) {
            for (int j = 0; j < X.getColumnDimension(); j++) {
                mat[i][j] = f_sigmoid(X.getEntry(i, j), deriv);
            }
        }
        return MatrixUtils.createRealMatrix(mat);
    }
    
    public double f_sigmoid(double X, boolean deriv){
        if(deriv == false){
            return 1 / (1 + Math.exp(-X));
        }else{
            return f_sigmoid(X, false)*(1 - f_sigmoid(X, false));
        }
    }


    public double f_softmax(double X){
        //Z = Math.sum(np.exp(X), axis=1)
        //Z = Z.reshape(Z.shape[0], 1)
        //return Math.exp(X) / Z
        return f_sigmoid(X, false);
    }
}
