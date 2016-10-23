//package ann;
///**
// * Activation function which enforces that output neurons have probability distribution (sum of all outputs is one)
// */
//public class SoftMax  {
//
//    private Layer layer;
//    private double totalLayerInput;
//
//    public SoftMax(final Layer layer) {
//        this.layer = layer;
//    }
//
//    public double getOutput(double netInput) {
//        totalLayerInput = 0;
//        for (Neuron neuron : layer.neurons) {
//            totalLayerInput += Math.exp(neuron.getNetInput());
//        }
//        output = Math.exp(netInput) / totalLayerInput;
//        return output;
//    }
//
//    public double getDerivative(double net) {
//        return 1d * (1d - output);
//    }
//}