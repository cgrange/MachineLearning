import java.util.Random;


public class Weight {
	private double value, lastChange;
	
	public Weight(double value){
		this.value = value;
		this.lastChange = 0;
	}
	
	public void delta(double deltaW){
		if(NeuralNet.getInstance(new Random()).useMomentum()){
			deltaW += lastChange*NeuralNet.getInstance(new Random()).getMomentum();
		}
		value += deltaW;
		lastChange = deltaW;
	}
	
	public double getValue(){
		return value;
	}
}
