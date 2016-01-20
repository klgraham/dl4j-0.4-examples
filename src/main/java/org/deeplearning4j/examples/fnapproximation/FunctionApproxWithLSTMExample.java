package org.deeplearning4j.examples.fnapproximation;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * This is a function approximation example. A better one than my first attempt.
 */
public class FunctionApproxWithLSTMExample
{
	public static void main( String[] args ) throws Exception {
		int inputLayerSize = 1;                     //Number of units in the input layer
		int outputLayerSize = 2;                    //Number of units in the output layer = number of predictions for each input
		int lstmLayerSize = 10;					    //Number of units in each GravesLSTM layer
		int numIterations = 5001;

		int nSamples = 10;

		double[] xx = new double[nSamples];
		double[] xx1 = new double[nSamples+1];
		double[] yy = new double[2 * nSamples];
		double a = 0.5*Math.PI / (double)nSamples;

		for (int i = 0; i < nSamples; i++) {
			xx[i] = (double)i * a;
			xx1[i] = xx[i];
		}
		xx1[nSamples] = a * (double)nSamples;
		for (int i = 0; i < 2*nSamples; i += 2) {
			yy[i] = f(xx1[i/2]);
			yy[i+1] = f(xx1[i/2 + 1]);
		}

		INDArray x = Nd4j.create(xx).reshape(nSamples,1);
		INDArray y = Nd4j.create(yy).reshape(nSamples,outputLayerSize);
		DataSet data = new DataSet(x, y);
//		System.out.println("x: \n" + x);
//		System.out.println("y: \n" + y);

		//Set up network configuration:
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.iterations(numIterations)
				.learningRate(0.1)
//			.rmsDecay(0.95)
				.seed(12345)
				.regularization(true)
				.l2(0.0)
				.list(3)
				.layer(0, new GravesLSTM.Builder().nIn(inputLayerSize).nOut(lstmLayerSize)
					.updater(Updater.ADAGRAD)
					.activation("tanh").weightInit(WeightInit.DISTRIBUTION)
					.dist(new UniformDistribution(-0.08, 0.08)).build())
				.layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
					.updater(Updater.ADAGRAD)
					.activation("tanh").weightInit(WeightInit.DISTRIBUTION)
					.dist(new UniformDistribution(-0.08, 0.08)).build())
				.layer(2, new RnnOutputLayer.Builder(LossFunction.MSE).activation("sigmoid")
//			.layer(2, new RnnOutputLayer.Builder(LossFunction.MSE).activation("softmax")
			         .updater(Updater.ADAGRAD).nIn(lstmLayerSize).nOut(outputLayerSize)
			         .weightInit(WeightInit.DISTRIBUTION)
	                 .dist(new UniformDistribution(-0.08, 0.08)).build())
				.pretrain(false).backprop(true)
				.build();
		
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
//		data.normalize();
		net.setListeners(new ScoreIterationListener(100));
		
		//Print the  number of parameters in the network (and for each layer)
		Layer[] layers = net.getLayers();
		int totalNumParams = 0;
		for( int i=0; i<layers.length; i++ ){
			int nParams = layers[i].numParams();
			System.out.println("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}
		System.out.println("Total number of network parameters: " + totalNumParams);
		net.fit(data);

//		INDArray yEstimate = net.output(x);
//		System.out.println("F1: " + net.f1Score(data));
//		INDArray error = Nd4j.std(y.sub(yEstimate));
//		System.out.println("y: " + y);
//		System.out.println("yEst: " + yEstimate);
//		System.out.println("Error= " + error);

//		double[] xxShift = new double[nSamples];
//		for (int i = 0; i < nSamples; i++) {
//			xxShift[i] = xx[i] + 0.05;
//		}
		double t1 = 1, t2 = -0.5, dt = a;
		double y1 = f(t1), y2 = f(t2), y1p = f(t1 + dt), y2p = f(t2 + dt);

		double[] x0 = new double[]{t1, t2};
		INDArray input = Nd4j.create(x0).reshape(2,1);

		INDArray output = net.output(input).reshape(2,2);
		INDArray expectedOutput = Nd4j.create(new double[]{y1, y1p, y2, y2p}).reshape(2,2);
		INDArray error = Transforms.abs(output.add(expectedOutput.mul(-1.0)));

		System.out.println("input:\n" + input);
		System.out.println("output:\n" + output);
		System.out.println("expected output:\n" + expectedOutput);
		System.out.println("prediction error:" + error);
	}

	static double f(double x) {
//		return x * x;
		return Math.sin(x);
	}


}