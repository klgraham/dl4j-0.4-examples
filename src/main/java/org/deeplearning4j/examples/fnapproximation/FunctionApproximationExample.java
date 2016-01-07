package org.deeplearning4j.examples.fnapproximation;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by klgraham on 9/16/15.
 */
public class FunctionApproximationExample {

    public static void main(String[] args) throws Exception {
        int numInputs = 1;
        int numOutputs = 1;
        int numHiddenNodes = 1000;
        int nSamples = 10;

        int seed = 123;
        int iterations = 100;
        float learningRate = 1e-3f;

        INDArray x = Nd4j.linspace(1, nSamples, nSamples).reshape(nSamples, 1);
        INDArray y = x.mul(x);
        DataSet dataSet = new DataSet(x, y);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed) // Seed to lock in weight initialization for tuning
                .iterations(iterations) // # training iterations predict/classify & backprop
                .learningRate(learningRate) // Optimization step size
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT) // Backprop method (calculate the gradients)
//                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .l2(2e-4).regularization(true)
                .list(2) // # NN layers (does not count input layer)
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation("tanh")
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(numHiddenNodes)
                        .nOut(numOutputs) // # output nodes
                        .activation("identity")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .backprop(true)
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.setListeners(new ScoreIterationListener(10));
        network.fit(dataSet);

        INDArray yEstimate = network.output(x);
        INDArray error = Nd4j.std(y.sub(yEstimate));

        System.out.println("error: " + error);
//        System.out.println("x: " + x);
//        System.out.println("y: " + y);

    }

}
