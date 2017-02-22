import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public abstract class Neuron {
	
	private List<Weight> weights;
	protected double output, error;
	Random rand;
	int numInputs;
	
	public Neuron(int numInputs, Random rand){
		this.rand = rand;
		weights = new ArrayList<Weight>();
		this.numInputs = numInputs;
	}
	
	private double randomWeight(){
		return (rand.nextDouble() - .5)*2;
	}
	
	private void initTestWeights(){
		int layer = NNFacade.layerOfHiddenNeuron(this);
		Weight w1, w2, bias;
		w1 = new Weight(1.0);
		w2 = new Weight(1.0);
		bias = new Weight(1.0);
		weights.add(w1);
		weights.add(w2);
		weights.add(bias);
	}
	
	public void initializeIncomingWeights(){
		if(NNFacade.testing()){
			initTestWeights();
		}
		else{
			for(int i = 0; i < numInputs; i++){
				Weight randomWeight = new Weight(randomWeight());
				weights.add(randomWeight);
			}
			Weight randomWeight = new Weight(randomWeight());
			weights.add(randomWeight); // for bias
		}
	}
	
	private double getNet(double[] inputs){
		double net = 0;
		int i;
		for(i = 0; i < inputs.length; i++){
			net += weights.get(i).getValue() * inputs[i];
		}
		net += weights.get(i).getValue()*1; // for bias;
		return net;
	}
	
	private double fNet(double net){
		return 1/(1+Math.pow(Math.E, -net));
	}
	
	public double activate(double[] inputs){
		double net = getNet(inputs);
		output = fNet(net);
		return output;
	}
	
	public void updateWeights(double[] inputs){
		//deltaWij = learningRate * outputI * errorJ
		double deltaWij;
		int i;
		for(i  = 0; i < weights.size()-1; i++){ // minus one for the bias
			double outputI;
			if(NNFacade.layerOfHiddenNeuron(this) == 0){
				outputI = inputs[i];
			}
			else{
				outputI = NNFacade.outputI(this, i);
			}
			deltaWij = NNFacade.getLearningRate() * outputI * error;
			weights.get(i).delta(deltaWij);
		}
		deltaWij = NNFacade.getLearningRate() * 1 * error;
		weights.get(i).delta(deltaWij); // for bias
	}
	
	public double fPrimeNet(){
		return output*(1-output);
	}
	
	public double getOutput(){
		return output;
	}
	
	public double getError(){
		return error;
	}
	
	public double getWeightFrom(int i){
		return weights.get(i).getValue();
	}
	
	public abstract double computeError(double target);
}
