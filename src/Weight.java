import java.util.Random;


public class Weight {
	private double value, lastChange;
	
	public Weight(double value){
		this.value = value;
		this.lastChange = 0;
	}
	
	public void delta(double deltaW){
		if(NNFacade.testing()){
			System.out.println("deltaW: " + deltaW);
		}
		value += deltaW;
		if(NeuralNet.getInstance(new Random()).useMomentum()){
			value += lastChange*NeuralNet.getInstance(new Random()).getMomentum();
		}
		lastChange = deltaW;
	}
	
	public double getValue(){
		return value;
	}
}
