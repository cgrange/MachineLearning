import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;


public class InstanceBasedLearner extends SupervisedLearner {
	
	int k = 3;
	boolean distanceWeighting = false;
	
	Matrix features;
	Matrix labels;
	TreeMap<Double, Double> distanceToClass;
	boolean outputNominal;
	
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		features.normalize();
		labels.normalize();
		Random rand = new Random();
		features.shuffle(rand, labels);
		distanceToClass = new TreeMap<Double, Double>();
		
		if(labels.valueCount(0) > 0) outputNominal = true;
		else outputNominal = false;
		
		Matrix trainingSet = new Matrix();
		trainingSet.loadArff("datasets/magicTelescopeTraining.arff");
		trainingSet.normalize();
		trainingSet.shuffle(rand);
		
		this.features = new Matrix(trainingSet, 0, 0, trainingSet.rows(), trainingSet.cols() - 1);
		this.labels = new Matrix(trainingSet, 0, trainingSet.cols()-1, trainingSet.rows(), 1);
		int originalSize = this.features.rows();
		double originalAccuracy = measureAccuracy(features, labels, new Matrix());
		System.out.println("original accuracy: " + originalAccuracy);
		System.out.println("original size: " + this.features.rows());
		
		double newAccuracy;
		do{//reduce matrix size by 5 percent
			this.features = new Matrix(this.features, (int)(this.features.rows()*.05), 0, (int)(this.features.rows()-this.features.rows()*.05), this.features.cols());
			this.labels = new Matrix(this.labels, (int)(this.labels.rows()*.05), 0, (int)(this.labels.rows()-this.labels.rows()*.05), this.labels.cols());
			newAccuracy = measureAccuracy(features, labels, new Matrix());
			System.out.println("new accuracy: " + newAccuracy);
			System.out.println("new size: " + this.features.rows());
		}while(originalAccuracy - newAccuracy < .02);
		System.out.println("final accuracy: " + newAccuracy);
		System.out.println("final size: " + this.features.rows());
		double ratio = (double)this.features.rows()/(double)originalSize;
		System.out.println((100 - (ratio*100)) + "% reduction in training set size");
		
	}
	
	private void getDistances(double[] features, double[] labels){
		distanceToClass.clear();
		for(int r = 0; r < this.features.rows(); r++){
			double[] row = this.features.row(r);
			double classValue = this.labels.row(r)[0];
			double distanceSquared = 0;
			
			for(int c = 0; c< row.length; c++){
				if(features[c] == Double.MAX_VALUE || row[c] == Double.MAX_VALUE){ // the value is unknown
					distanceSquared += 1; //maximum distance in a normalized set
				}
				else if(this.features.valueCount(c) != 0){// the value is  nominal
					if(row[c] != features[c]){
						distanceSquared += 1; //maximum distance in a normalized set (I'm assuming that different classes are entirely unique)
					}
					else distanceSquared += 0;
				}
				else{// the value is continuous
					distanceSquared += Math.pow((row[c] - features[c]), 2);
				}
			}
			double distance = Math.sqrt(distanceSquared);
			distanceToClass.put(distance, classValue);
		}	
	}
	
	private double majorityClass(){
		Map<Double, Integer> classValueToInstances = new HashMap<Double, Integer>();
		Iterator<Double> it = distanceToClass.keySet().iterator();
		for(int i = 0; i < k; i++){
			double distanceKey = it.next();
			double outputValue = distanceToClass.get(distanceKey);
			if(classValueToInstances.containsKey(outputValue)){
				int instances = classValueToInstances.get(outputValue);
				classValueToInstances.put(outputValue, instances+1);
			}
			else{
				classValueToInstances.put(outputValue, 1);
			}
		}
		Iterator<Double> it2 = classValueToInstances.keySet().iterator();
		int maxInstances = 0;
		double majorityClass = -1.0;
		while(it2.hasNext()){
			double key = it2.next();
			int instances = classValueToInstances.get(key);
			if(instances > maxInstances){
				maxInstances = instances;
				majorityClass = key;
			}
		}
		return majorityClass;
	}
	
	private TreeMap<Double, Double> reverse(Map<Double, Double> original){
		TreeMap<Double, Double> reverse = new TreeMap<Double, Double>();
		Iterator<Double> keyIt = original.keySet().iterator();
		while(keyIt.hasNext()){
			double key = keyIt.next();
			double value = original.get(key);
			reverse.put(value, key);
		}
		return reverse;
	}
	
	private double distanceWeightedClass(){
		Map<Double, Double> classToDistanceWeighted = new HashMap<Double, Double>();
		Iterator<Double> it = distanceToClass.keySet().iterator();
		for(int i = 0; i < k; i++){
			double distanceKey = it.next();
			double outputClass = distanceToClass.get(distanceKey);
			double distanceWeighted = 1/Math.pow(distanceKey, 2);
			if(classToDistanceWeighted.containsKey(outputClass)){
				double curr = classToDistanceWeighted.get(outputClass);
				classToDistanceWeighted.put(outputClass, curr+distanceWeighted);
			}
			else{
				classToDistanceWeighted.put(outputClass, distanceWeighted);
			}
		}
		TreeMap<Double, Double> distanceWeightedToClass = reverse(classToDistanceWeighted);
		return distanceWeightedToClass.descendingMap().firstEntry().getValue();
	}
	
	private double meanOutputKNN(){
		Iterator<Double> keyIt = distanceToClass.keySet().iterator();
		double total = 0;
		for(int i = 0; i < k; i++){
			double distanceKey = keyIt.next();
			double outputValue = distanceToClass.get(distanceKey);
			total += outputValue;
		}
		return total/k;
	}
	
	private double distanceWeightedRegression(){
		Iterator<Double> keyIt = distanceToClass.keySet().iterator();
		double weightedTotal = 0;
		double sumWeight = 0;
		for(int i = 0; i < k; i++){
			double distanceKey = keyIt.next();
			double weight = 1/Math.pow(distanceKey, 2);
			double classValue = distanceToClass.get(distanceKey);
			weightedTotal += classValue * weight;
			sumWeight += weight;
		}
		return weightedTotal/sumWeight;
	}
	
	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		getDistances(features, labels);
		if(outputNominal){
			if(distanceWeighting){//each output class = sum((1/distance)^2) for each instance of that class in knn. Class with greatest value is the output
				labels[0] = distanceWeightedClass();
			}
			else{// output = majority class of knn 
				labels[0] = majorityClass();
			}
		}
		else{// output is continuous
			if(distanceWeighting){// output = sum(output*weight)/sum(weight);
				labels[0] = distanceWeightedRegression();
			}
			else{// output = average of outputs of knn
				labels[0] = meanOutputKNN();
			}
		}
	}

	@Override
	public void mseTest(Matrix features, Matrix labels) {
	}

}
