import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class NNFacade {
	/**
	 * @param n
	 * @return the hidden layer that contains this neuron or -1 if it is not a hidden neuron
	 */
	public static boolean testing(){
		return NeuralNet.getInstance(new Random()).testing;
	}
	
	public static int layerOfHiddenNeuron(Neuron n){
		int layer = -1;
		for(int i = 0; i < NeuralNet.getInstance(new Random()).hiddenLayers.size(); i++){
			if(NeuralNet.getInstance(new Random()).hiddenLayers.get(i).contains(n)){
				layer = i;
			}
		}
		return layer;
	}
	
	public static double outputI(Neuron nJ, int i){
		double outputI;
		int previousLayer = layerOfHiddenNeuron(nJ)-1;
		if(previousLayer > -1){
			outputI = NeuralNet.getInstance(new Random()).hiddenLayers.get(previousLayer).get(i).getOutput();
		}
		else{
			int lastHiddenLayer = NeuralNet.getInstance(new Random()).hiddenLayers.size()-1;
			outputI = NeuralNet.getInstance(new Random()).hiddenLayers.get(lastHiddenLayer).get(i).getOutput();
		}
		return outputI;
	}
	
	public static double getLearningRate(){
		return NeuralNet.getInstance(new Random()).getLearningRate();
	}
	
	public static List<ArrayList<Neuron>> getHiddenLayers(){
		return NeuralNet.getInstance(new Random()).hiddenLayers;
	}
	
	public static List<Neuron> getOutputNeurons(){
		return NeuralNet.getInstance(new Random()).outputNeurons;
	}
	
	public static double[] getErrorsK(Neuron nJ){
		double[] errorsK;
		int j = layerOfHiddenNeuron(nJ);
		int k = j+1;
		if(getHiddenLayers().size()-1 == j){// this neuron is in the last layer of hidden  nodes
			errorsK = new double[getOutputNeurons().size()];
			for(int n = 0; n < getOutputNeurons().size(); n++){
				errorsK[n] = getOutputNeurons().get(n).getError();
			}
		}
		else{
			List<Neuron> layerK = getHiddenLayers().get(k);
			errorsK = new double[layerK.size()];
			for(int n = 0; n < layerK.size(); n++){
				errorsK[n] = layerK.get(n).getError();
			}
		}
		return errorsK;
	}
	
	public static double[] getWeightsJK(Neuron nJ){
		double[] weightsJK;
		int j = layerOfHiddenNeuron(nJ);
		int k = j+1;
		int nJindex = getHiddenLayers().get(j).indexOf(nJ);
		if(getHiddenLayers().size()-1 == j){// this neuron is in the last layer of hidden  nodes
			weightsJK = new double[getOutputNeurons().size()];
			for(int n = 0; n < getOutputNeurons().size(); n++){
				weightsJK[n] = getOutputNeurons().get(n).getWeightFrom(nJindex);
			}
		}
		else{
			List<Neuron> layerK = getHiddenLayers().get(k);
			weightsJK = new double[layerK.size()];
			for(int n = 0; n < layerK.size(); n++){
				weightsJK[n] = layerK.get(n).getWeightFrom(nJindex);
			}
		}
		return weightsJK;
	}
	
}
