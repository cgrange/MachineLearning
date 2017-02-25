import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public abstract class Neuron {
	
	protected List<Weight> weights;
	protected double output, error;
	Random rand;
	int numInputs;
	
	public Neuron(int numInputs, Random rand){
		this.rand = rand;
		weights = new ArrayList<Weight>();
		this.numInputs = numInputs;
	}
	
	public Neuron(int numInputs, Random rand, List<Weight> weights){
		this.rand = rand;
		this.weights = new ArrayList<Weight>();
		for(Weight w : weights){
			this.weights.add(new Weight(w.getValue()));
		}
		this.numInputs = numInputs;
		
	}
	
	private double randomWeight(){
		return (rand.nextDouble() - .5)*2;
	}
	
	private void initTestWeights(){
		Weight w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12;
		w0 = new Weight(0.02);
		w1 = new Weight(-0.01);
		w2 = new Weight(0.03);
		w3 = new Weight(0.02);
		w4 = new Weight(-0.01);
		w5 = new Weight(-0.03);
		w6 = new Weight(0.03);
		w7 = new Weight(0.01);
		w8 = new Weight(0.04);
		w9 = new Weight(-0.02);
		w10 = new Weight(-0.02);
		w11 = new Weight(0.03);
		w12 = new Weight(0.02);
		int layer = NNFacade.layerOfHiddenNeuron(this);
		if(layer != -1){
			int idx = NNFacade.getHiddenLayers().get(layer).indexOf(this);
			if(idx == 0){
				weights.add(w5);
				weights.add(w6);
				weights.add(w4);
			}
			else if(idx == 1){
				weights.add(w8);
				weights.add(w9);
				weights.add(w7);
			}
			else if(idx == 2){
				weights.add(w11);
				weights.add(w12);
				weights.add(w10);
			}
			else{
				System.out.println("You're doing it wrong");
				System.exit(0);
			}
		}
		else{ // it's the output node
			weights.add(w1);
			weights.add(w2);
			weights.add(w3);
			weights.add(w0);
		}
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
	
	public List<Weight> getWeights(){
		return this.weights;
	}
	
	public abstract double computeError(double target);
	
	public abstract void printWeights();
	public abstract void printOutput();
	public abstract void printError();
}
