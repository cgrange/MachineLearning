import java.util.Random;


public class HiddenNeuron extends Neuron {

	public HiddenNeuron(int numInputs, Random rand) {
		super(numInputs, rand);
		// TODO Auto-generated constructor stub
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
