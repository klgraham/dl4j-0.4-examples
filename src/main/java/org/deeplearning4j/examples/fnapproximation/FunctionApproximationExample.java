package org.deeplearning4j.examples.fnapproximation;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by klgraham on 9/16/15.
 */
public class FunctionApproximationExample
{
	public static void main(String[] args) throws Exception
	{
		int numInputs = 1;
		int numOutputs = 1;
		int nSamples = 10;
		int iterations = 200;
//        int numHiddenNodes = 1000;
		int seed = 123;
		float learningRate = 1e-3f;

		INDArray x = Nd4j.linspace(1, nSamples, nSamples).reshape(nSamples, 1);
		INDArray y = x.mul(x);
		DataSet dataSet = new DataSet(x, y);

		int[] hiddenNodes = {20, 50, 75, 100, 200, 500, 750, 1000,
		                     1500, 2000, 2500, 3000, 3500, 4000, 5000,
		                     7500, 10000, 15000, 25000, 50000, 75000, 100000};

		StringBuilder sb = new StringBuilder();
		for (int numHiddenNodes : hiddenNodes)
		{
			MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
					.seed(seed) // Seed to lock in weight initialization for tuning
					.iterations(iterations) // # training iterations predict/classify & backprop
					.learningRate(learningRate) // Optimization step size
					.optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT) // Backprop method (calculate the gradients)
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
//            network.setListeners(new ScoreIterationListener(10));
			dataSet.normalize();
			network.fit(dataSet);

			INDArray yEstimate = network.output(x);
			INDArray error = Nd4j.std(y.sub(yEstimate));

			sb.append(numHiddenNodes + "," + error);
			System.out.println(numHiddenNodes + "," + error);
			sb.append("\n");
		}
		System.out.println("*****************");
		System.out.println(sb.toString());
	}

}
