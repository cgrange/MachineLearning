import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Random;


public class Perceptron extends SupervisedLearner {
	Random rand;
	double learnRate;
	ArrayList<Double> weights1;
	ArrayList<Double> weights2;
	ArrayList<Double> weights3;
	
	public Perceptron(Random rand){
		this.rand = rand;
		learnRate = .01;
		weights1 = new ArrayList<Double>();
		weights2 = new ArrayList<Double>();
		weights3 = new ArrayList<Double>();
	}

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		features.normalize();
  		initializeWeights(features.cols());
		List<Double> last5accuracyMeasurements = new LinkedList<Double>();
		double accuracy;
		int epochs = 0;
		do{
			features.shuffle(rand, labels);
			for(int r = 0; r < features.rows(); r++){
				int output1 = processInputs(features, r, weights1);
				int output2 = processInputs(features, r, weights2);
				int output3 = processInputs(features, r, weights3);
				adjustWeights1(r, features, labels, output1);
				adjustWeights2(r, features, labels, output2);
				adjustWeights3(r, features, labels, output3);
			}
			epochs++;
			Matrix confusion = new Matrix();
			accuracy = measureAccuracy(features, labels, confusion);
			last5accuracyMeasurements.add(accuracy);
			System.out.println(epochs + ", " + accuracy);
		}while(accuracyIsImproving(last5accuracyMeasurements));
		outputWeights();
	}
	
	private boolean accuracyIsImproving(List<Double> last5accuracyMeasurements){
		if(last5accuracyMeasurements.size() < 5)return true; // will go through at least 5 epochs;
		double min, max, current;
		max = last5accuracyMeasurements.get(0);
		min = last5accuracyMeasurements.get(0);
		current = last5accuracyMeasurements.get(0);
		for(int i = 0; i < last5accuracyMeasurements.size(); i++){ // min and max accuracy over last 5
			current = last5accuracyMeasurements.get(i);
			if(current < min){
				min = current;
			}
			if(current > max){
				max = current;
			}
		}
		if(current == max){
			return false;
		}
		return true;
	}
	
	private void initializeWeights(int numInputs){
		for(int c = 0; c < numInputs; c++){
			weights1.add(0.0/*randomNum()*/);
			weights2.add(0.0/*randomNum()*/);
			weights3.add(0.0/*randomNum()*/);
		}
		weights1.add(0.0/*randomNum()*/);// for bias
		weights2.add(0.0/*randomNum()*/);// for bias
		weights3.add(0.0/*randomNum()*/);// for bias
		//outputWeights();
	}
	
	private void outputWeights(){
		for(int i = 0; i < weights1.size(); i++){
			System.out.println("weights1 " + i + ": " + weights1.get(i));
		}
		for(int i = 0; i < weights1.size(); i++){
			System.out.println("weights2 " + i + ": " + weights2.get(i));
		}
		for(int i = 0; i < weights1.size(); i++){
			System.out.println("weights3 " + i + ": " + weights3.get(i));
		}
	}
	
	private void adjustWeights1(int row, Matrix features, Matrix labels, int output){
		int c;
		int target = (int) labels.get(row, 0);
		if(target == 0)target = 1;
		else target = 0;
		for(c = 0; c < features.cols(); c++){
			double input = features.get(row, c);
			double deltaW = learnRate*(target - output)*input;
			double oldWeight = weights1.get(c);
			double newWeight = oldWeight + deltaW;
			weights1.set(c, newWeight);
		}
		// ************* adjust bias ******************
		double input = 1; 
		double deltaW = learnRate*(target - output)*input;
		double oldWeight = weights1.get(c);
		double newWeight = oldWeight + deltaW;
		weights1.set(c, newWeight);
		//outputWeights();
	}
	
	private void adjustWeights2(int row, Matrix features, Matrix labels, int output){
		int c;
		int target = (int) labels.get(row, 0);
		if(target == 1)target = 1;
		else target = 0;
		for(c = 0; c < features.cols(); c++){
			double input = features.get(row, c);
			double deltaW = learnRate*(target - output)*input;
			double oldWeight = weights2.get(c);
			double newWeight = oldWeight + deltaW;
			weights2.set(c, newWeight);
		}
		// ************* adjust bias ******************
		double input = 1; 
		double deltaW = learnRate*(target - output)*input;
		double oldWeight = weights2.get(c);
		double newWeight = oldWeight + deltaW;
		weights2.set(c, newWeight);
		//outputWeights();
	}
	
	private void adjustWeights3(int row, Matrix features, Matrix labels, int output){
		int c;
		int target = (int) labels.get(row, 0);
		if(target == 2)target = 1;
		else target = 0;
		for(c = 0; c < features.cols(); c++){
			double input = features.get(row, c);
			double deltaW = learnRate*(target - output)*input;
			double oldWeight = weights3.get(c);
			double newWeight = oldWeight + deltaW;
			weights3.set(c, newWeight);
		}
		// ************* adjust bias ******************
		double input = 1; 
		double deltaW = learnRate*(target - output)*input;
		double oldWeight = weights3.get(c);
		double newWeight = oldWeight + deltaW;
		weights3.set(c, newWeight);
		//outputWeights();
	}
	
	private int processInputs(Matrix features, int row, ArrayList<Double> weights){
		double input, result;
		result = 0;
		int c;
		for(c = 0; c < features.cols(); c++){
			input = features.get(row, c);
			result += weights.get(c) * input;
		}
		input = 1;// for bias
		result += weights.get(c) * input; // bias
		if(result > 0){
			return 1;
		}
		return 0;
	}
	
	public void generateValues(){
		for(int n = 0; n < 8; n++){
			double x, y;
			x = randomNum();
			y = randomNum();
			//System.out.println(x + "," + y + ",");
		}
	}
	
	private double randomNum(){
		double randomNum = (Math.random() -.5)*2;
		return randomNum;
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		double result1, result2, result3;
		result1 = 0;
		result2 = 0;
		result3 = 0;
		int i;
		for(i = 0; i < features.length; i++){
			result1 += features[i]*weights1.get(i);
			result2 += features[i]*weights2.get(i);
			result3 += features[i]*weights3.get(i);
		}
		result1 += weights1.get(i);// for the bias no need to multiply by input since it's always 1;
		result2 += weights2.get(i);
		result3 += weights3.get(i);
		if(result1 > result2 && result1 > result3){
			labels[0] = 0;
		}
		else if(result2 > result1 && result2 > result3){
			labels[0] = 1;
		}
		else{
			labels[0] = 2;
		}
	}

	@Override
	public void mseTest(Matrix features, Matrix labels) {
		// TODO Auto-generated method stub
		
	}

}
