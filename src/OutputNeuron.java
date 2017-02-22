import java.util.Random;


public class OutputNeuron extends Neuron {

	public OutputNeuron(int numInputs, Random rand) {
		super(numInputs, rand);
		// TODO Auto-generated constructor stub
	}

	@Override
	public double computeError(double target) {
		error = (target - output) * fPrimeNet();
		return error;
	}

}
