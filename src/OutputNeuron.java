import java.util.List;
import java.util.Random;


public class OutputNeuron extends Neuron {

	public OutputNeuron(int numInputs, Random rand) {
		super(numInputs, rand);
	}

	public OutputNeuron(int numInputs, Random random, List<Weight> weights) {
		super(numInputs, random, weights);
	}

	@Override
	public double computeError(double target) {
		error = (target - output) * fPrimeNet();
		return error;
	}

}
