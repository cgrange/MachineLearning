import java.util.List;
import java.util.Random;


public class OutputNeuron extends Neuron {

	public OutputNeuron(int numInputs, Random rand) {
		super(numInputs, rand);
	}

	public OutputNeuron(int numInputs, Random random, List<Weight> weights) {
		super(numInputs, random, weights);
	}
	
	public void printWeights(){
		System.out.print("w_0=" + weights.get(3).getValue() + ", ");
		System.out.print("w_1=" + weights.get(0).getValue() + ", ");
		System.out.print("w_2=" + weights.get(1).getValue() + ", ");
		System.out.print("w_3=" + weights.get(2).getValue() + ", ");
	}

	@Override
	public double computeError(double target) {
		error = (target - output) * fPrimeNet();
		return error;
	}

	@Override
	public void printOutput() {
		System.out.print("o_0=" + output + ", ");
	}

	@Override
	public void printError() {
		System.out.print("e_0=" + error + ", ");
	}

}
