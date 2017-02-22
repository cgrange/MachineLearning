import java.util.List;
import java.util.Random;


public class HiddenNeuron extends Neuron {

	public HiddenNeuron(int numInputs, Random rand) {
		super(numInputs, rand);
	}

	public HiddenNeuron(int numInputs, Random myRand, List<Weight> weights) {
		super(numInputs, myRand, weights);
	}

	@Override
	public double computeError(double target) {
		// ERRORj = sum ERRORk * WEIGHTjk * fPrimeNetj for all K
		double errorJ = 0;
		double[] errorsK = NNFacade.getErrorsK(this);
		double[] weightsJK = NNFacade.getWeightsJK(this);
		for(int i = 0; i < errorsK.length; i++){
			errorJ += errorsK[i] * weightsJK[i] * fPrimeNet();
		}
		error = errorJ;
		return error;
	}

}
