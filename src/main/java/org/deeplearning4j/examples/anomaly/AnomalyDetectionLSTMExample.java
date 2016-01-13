package org.deeplearning4j.examples.anomaly;

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

/**GravesLSTM Character modelling example
 * @author Alex Black

   Example: Train a LSTM RNN to generates text, one character at a time.
	This example is somewhat inspired by Andrej Karpathy's blog post,
	"The Unreasonable Effectiveness of Recurrent Neural Networks"
	http://karpathy.github.io/2015/05/21/rnn-effectiveness/
	
	Note that this example has not been well tuned - better performance is likely possible with better hyperparameters
	
	Some differences between this example and Karpathy's work:
	- The LSTM architectures appear to differ somewhat. GravesLSTM has peephole connections that
	  Karpathy's char-rnn implementation appears to lack. See GravesLSTM javadoc for details.
	  There are pros and cons to both architectures (addition of peephole connections is a more powerful
	  model but has more parameters per unit), though they are not radically different in practice.
	- Karpathy uses truncated backpropagation through time (BPTT) on full character
	  sequences, whereas this example uses standard (non-truncated) BPTT on partial/subset sequences.
	  Truncated BPTT is probably the preferred method of training for this sort of problem, and is configurable
      using the .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength().tBPTTBackwardLength() options
	  
	This example is set up to train on the Complete Works of William Shakespeare, downloaded
	 from Project Gutenberg. Training on other text sources should be relatively easy to implement.
 */
public class AnomalyDetectionLSTMExample
{
	public static void main( String[] args ) throws Exception {
		int inputLayerSize = 1;                     //Number of units in the input layer
		int outputLayerSize = 2;                    //Number of units in the output layer = number of predictions for each input
		int lstmLayerSize = 10;					    //Number of units in each GravesLSTM layer
		int numIterations = 4001;

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
//				.regularization(true)
//				.l2(0.001)
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
		INDArray x1 = Nd4j.create(new double[]{1, 0.5}).reshape(2,1);
		INDArray y1 = net.output(x1);
		System.out.println("x1: " + x1);
		System.out.println("y1: " + y1);
	}

	static double f(double x) {
//		return x * x;
		return Math.sin(x);
	}


}