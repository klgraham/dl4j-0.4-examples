package org.deeplearning4j.examples.anomaly;

import java.util.List;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

/**
 * A DataSetIterator for time-series data
 * 
 * Given a time-series X = {x(1), x(2), ..., x(N)}, where
 * x(t) = {x(t)_1, ..., x(t)_M}, this will iterate through the x(t)
 *
 * @author kgraham
 *         1/15/16
 */
public class TimeSeriesIterator implements DataSetIterator
{
	private double[] sequence;              // the time series, as a list
	private int numTrainingExamples;            // Use the first numTrainingExamples data points to train with
	private int numExamplesToFetch;             // Number of data points to use as retrieve
	private int timestep = 0;                   // Number of elements of the time series seen so far, the time-step variable
	private int numberFutureValues;             // Number of future values to return with an element of the time-series

	public TimeSeriesIterator(final double[] sequence, final int numTrainingExamples, final int numberFutureValues)
	{
		this.sequence = sequence;
		this.numTrainingExamples = numTrainingExamples;
		this.numberFutureValues = numberFutureValues;
	}


	/**
	 * If this is time-step t, return the next element of the time-series, x(t), and
	 * the next numberFutureValues in the series also, y(t) = {x(t+1), ..., x(t+numberFutureValues)}.
	 * @return DataSet(x, y)
	 */
	@Override
	public DataSet next()
	{
		// if there are enough timesteps left, get the next few values
		if (hasNext())
		{
			double x = sequence[timestep];
			double[] nextX = new double[numberFutureValues];
			for (int l = 0; l < numberFutureValues; l++)
			{
				nextX[l] = sequence[timestep + l + 1];
			}

			INDArray xx = Nd4j.create(new double[]{x});
			INDArray yy = Nd4j.create(nextX).reshape(1, numberFutureValues);
			return new DataSet(xx, yy);
		}
		throw new IllegalArgumentException("Not enough timesteps left.");
	}

	/**
	 *
	 * @param i
	 * @return
	 */
	@Override
	public DataSet next(final int i)
	{
		return null;
	}

	@Override
	public int totalExamples()
	{
		return 0;
	}

	@Override
	public int inputColumns()
	{
		return 0;
	}

	@Override
	public int totalOutcomes()
	{
		return 0;
	}

	@Override
	public void reset()
	{

	}

	@Override
	public int batch()
	{
		return 0;
	}

	@Override
	public int cursor()
	{
		return 0;
	}

	@Override
	public int numExamples()
	{
		return 0;
	}

	@Override
	public void setPreProcessor(final DataSetPreProcessor dataSetPreProcessor)
	{

	}

	@Override
	public List<String> getLabels()
	{
		return null;
	}

	@Override
	public boolean hasNext()
	{
		return (timestep + numberFutureValues) <= sequence.length;
	}
}
