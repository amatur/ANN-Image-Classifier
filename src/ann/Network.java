package ann;

import java.util.ArrayList;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class Network {

    Layer[] layers;
    int minibatchsize;
    ArrayList<Example> list;
    int numClasses;
    int numFeatures;

    public Network(ArrayList<Example> list, int[] layers, int minibatchsize) {
        //this.numLayers = layers.length;
        this.numFeatures = layers[0];
        this.numClasses = layers[layers.length - 1];
        this.minibatchsize = minibatchsize;
        this.list = list;
        this.layers = new Layer[layers.length];

        /**
         * *** input layer (0), hidden layers, output layer (numLayers-1)***
         */
        for (int i = 0; i < layers.length; i++) {
            // input layer
            if (i == 0) {
                System.out.println("Initializing input layer with size " + layers[i]);

                //add bias
                this.layers[0] = new Layer(layers[0] + 1, layers[1], true, false, minibatchsize);
            } // output layer
            else if (i == layers.length - 1) {
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

    RealMatrix forwardPropagate(ArrayList<Example> list) {
        //number of examples, n
        int n = list.size();

        //number of features + 1 for bias, m
        int m = this.numFeatures + 1;

        double[][] nmMat = new double[n][m];
        for (int i = 0; i < n; i++) {
            Example ex = list.get(i);
            for (int j = 0; j < m - 1; j++) {
                nmMat[i][j] = ex.features.getEntry(j);
            }
            //add bias
            for (int j = m - 1; j < m; j++) {
                nmMat[i][j] = 1;
            }
        }
        //X => n * m matrix -> example vs. features
        RealMatrix X = MatrixUtils.createRealMatrix(nmMat);
        this.layers[0].Z = X;   //input layer

       // System.out.println("input layer Z");
        //        System.out.println(this.layers[0].Z );
        //       System.out.println("");
        //
        //System.out.println("lay0 Z+" + this.layers[0].Z.getRowDimension());;
        //System.out.println(this.layers[0].Z.getColumnDimension());;
        for (int i = 0; i < layers.length - 1; i++) {
            // System.out.println(i);
            this.layers[i + 1].S = this.layers[i].forwardPropagate();
            // System.out.println("layer "+ (i+1) );
        }
        return this.layers[layers.length - 1].forwardPropagate();

    }

    public void backPropagate(RealMatrix yhat, RealMatrix labelMatrix) {
        this.layers[layers.length - 1].D = yhat.subtract(labelMatrix).transpose(); //row by row classes, column by column examples
        for (int i = layers.length - 2; i > 0; i--) {  //start from last hidden layer upto 1st hidden layer
            //We do not calculate deltas for the bias values
            RealMatrix W_nobias = this.layers[i].W.getSubMatrix(0, this.layers[i].W.getRowDimension() - 2, 0, this.layers[i].W.getColumnDimension() - 1);
            // System.out.println(W_nobias.getRowDimension());
            // System.out.println(W_nobias.getColumnDimension());
            this.layers[i].D = W_nobias.multiply(this.layers[i + 1].D);
          //  System.out.printf("D: %d %d\n ", this.layers[i].D.getRowDimension(), this.layers[i].D.getColumnDimension());
            //  System.out.printf("F: %d %d\n ", this.layers[i].Fp.getRowDimension(), this.layers[i].Fp.getColumnDimension());

            //column traversing => examples one by one
            for (int j = 0; j < this.layers[i].D.getColumnDimension(); j++) {
                RealVector col1 = this.layers[i].D.getColumnVector(j);
                RealVector col2 = this.layers[i].Fp.getColumnVector(j);
                this.layers[i].D.setColumnVector(j, col1.ebeMultiply(col2));
            }

        }
    }

    public void updateWeights(double eta) {
        //except output layer
        for (int i = 0; i < this.layers.length - 1; i++) {
            RealMatrix temp = this.layers[i + 1].D.multiply(this.layers[i].Z);
            RealMatrix W_grad = temp.transpose().scalarMultiply(-eta);
            this.layers[i].W = this.layers[i].W.add(W_grad);
        }
    }

    public void evaluate() {
        int numEpochs = 4000;
        double eta = 0.05;

        RealMatrix output = null;

        System.out.printf("Training for %d epochs... \n", numEpochs);
        for (int t = 0; t < numEpochs; t++) {
            //System.out.printf("Epoch %d: \n", t);

            
            /*** Step 1: train ***/
            for (int i = 0; i < list.size(); i = i + minibatchsize) {

                ArrayList<Example> sublist = new ArrayList<>();
                RealMatrix labelMatrix = MatrixUtils.createRealMatrix(minibatchsize, numClasses);

                for (int j = 0; j < minibatchsize; j++) {
                    labelMatrix.setRow(j, list.get(i + j).outputs.toArray());
                    sublist.add(list.get(i + j));
                }
                //System.out.println(sublist);
                //break;

                output = this.forwardPropagate(sublist); //yhat
                this.backPropagate(output, labelMatrix);
                this.updateWeights(eta);
                //System.out.printf("BATCH START\n");
                
                //System.out.println(output);
                //for (int j = 0; j < sublist.size(); j++) {
                //    System.out.printf("example %d : [%d E, %d A], \n", i + j, Example.getLabel(output.getRowVector(j)), sublist.get(j).getLabel());
                //}
                
                /*
                System.out.printf("Act: [");
                for (int j = 0; j < sublist.size(); j++) {
                    System.out.printf("%d, ", sublist.get(j).getLabel());
                }
                System.out.printf("]\n");

                System.out.printf("Est [");
                for (int j = 0; j < sublist.size(); j++) {
                    System.out.printf("%d, ", Example.getLabel(output.getRowVector(j)));
                }
                System.out.printf("]\n");
                System.out.printf("BATCHEND\n");
                */
                //return;
                /*
                
                 //System.out.println(output);
                 for (int j = 0; j < sublist.size(); j++) {
                 System.out.printf("example %d is classified %d, where actually %d\n\n", i + j, Example.getLabel(output.getRowVector(j)), sublist.get(j).getLabel());
                 }
                 */
            }
            
            
            
            /*** Step 2: test on train data ***/
            int correct = 0;
            int total = list.size();
            
            
            for (int i = 0; i < list.size(); i++) {
                ArrayList<Example> sublist = new ArrayList<>();
                sublist.add(list.get(i));
                RealMatrix subOutput = this.forwardPropagate(sublist);
                int actualLabel = sublist.get(0).getLabel();
                int estimatedLabel = Example.getLabel(subOutput.getRowVector(0));
                //System.out.printf("Act %d Est %d\n", actualLabel, estimatedLabel);
                 if (actualLabel == estimatedLabel){
                     
                     correct++;
                 }
                
            }
            double accuracy = correct*1.0 /total * 100.0;
            
            //System.out.printf("Epoch \t Correct \t Total \t Accuracy \t Eta \n ");
            System.out.printf("%d \t %d \t %d \t %f \t %f\n", t, correct, total, accuracy, eta);
            
            //return;
        }
    }

}
