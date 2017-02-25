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
	
	public void printWeights(){
		String startString0 = "nonesense";
		String startString1 = "nonesense";
		String startString2 = "nonesense";
		if(NNFacade.getHiddenLayers().get(0).indexOf(this) == 0){
			startString0 = "w_4=";
			startString1 = "w_5=";
			startString2 = "w_6=";
		}
		else if(NNFacade.getHiddenLayers().get(0).indexOf(this) == 1){
			startString0 = "\nw_7=";
			startString1 = "w_8=";
			startString2 = "w_9=";
		}
		else if(NNFacade.getHiddenLayers().get(0).indexOf(this) == 2){
			startString0 = "w_10=";
			startString1 = "w_11=";
			startString2 = "w_12=";
		}
		else{
			NNFacade.getHiddenLayers().get(-1);
		}
		System.out.print(startString0 + weights.get(2).getValue() + ", ");
		System.out.print(startString1 + weights.get(0).getValue() + ", ");
		System.out.print(startString2 + weights.get(1).getValue() + ", ");
	}

	@Override
	public void printOutput() {
		if(NNFacade.getHiddenLayers().get(0).indexOf(this) == 0){
			System.out.print("o_1=" + output);
		}
		else if(NNFacade.getHiddenLayers().get(0).indexOf(this) == 1){
			System.out.print("o_2=" + output);
		}
		else if(NNFacade.getHiddenLayers().get(0).indexOf(this) == 2){
			System.out.print("o_3=" + output);
		}
		else{
			NNFacade.getHiddenLayers().get(9999999);
			// you shouldn't be printing unless its for that one test case
		}
	}

	@Override
	public void printError() {
		String startString = "nonesense";
		if(NNFacade.getHiddenLayers().get(0).indexOf(this) == 0){
			startString = "e_1=";
		}
		else if(NNFacade.getHiddenLayers().get(0).indexOf(this) == 1){
			startString = "e_2=";
		}
		else if(NNFacade.getHiddenLayers().get(0).indexOf(this) == 2){
			startString = "e_3=";
		}
		else{
			NNFacade.getHiddenLayers().get(9999999);
			// you shouldn't be printing unless its for that one test case
		}
		System.out.print(startString + error + ", ");
	}

}
