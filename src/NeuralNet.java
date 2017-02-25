import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public class NeuralNet extends SupervisedLearner {
	
	List<ArrayList<Neuron>> hiddenLayers;
	List<Neuron> outputNeurons;
	private double learningRate, momentum;
	private boolean useMomentum;
	private int stoppingCriteria;
	Random rand, myRand;
	boolean testing;
	
	private class BestSolutionSoFar{
		private List<ArrayList<Neuron>> hiddenLayers;
		private List<Neuron> outputNeurons;
		private double mseValidation, mseTraining, accuracy;
		private int epochsWithoutImprovement;
		
		public BestSolutionSoFar(){
			mseValidation = 1;
			mseTraining = 1;
			epochsWithoutImprovement = 0;
		}

		public boolean hasImprovedOverLastNEpochs(double mseValidation, double mseTraining, double accuracy){
			if(mseValidation < this.mseValidation){
				this.mseValidation = mseValidation;
				this.mseTraining = mseTraining;
				this.accuracy = accuracy;
				//deep copy
				this.hiddenLayers = new ArrayList<ArrayList<Neuron>>();
				for(List<Neuron> layer : NeuralNet.this.hiddenLayers){
					ArrayList<Neuron> deepLayer = new ArrayList<Neuron>();
					for(Neuron n : layer){
						Neuron deepN = new HiddenNeuron(n.numInputs, myRand, n.getWeights());
						deepLayer.add(deepN);
					}
					this.hiddenLayers.add(deepLayer);
				}
				this.outputNeurons = new ArrayList<Neuron>();
				for(Neuron n : NeuralNet.this.outputNeurons){
					Neuron deepN = new OutputNeuron(n.numInputs, new Random(), n.getWeights());
					this.outputNeurons.add(deepN);
				}
				epochsWithoutImprovement = 0;
			}
			else{
				epochsWithoutImprovement++;
			}
			if(epochsWithoutImprovement == NeuralNet.this.stoppingCriteria){
				return false;
			}
			else{
				return true;
			}
		}
		
		public double getMseValidation(){
			return mseValidation;
		}
		
		public double getMseTraining(){
			return mseTraining;
		}
		
		public double getAccuracy(){
			return accuracy;
		}
		
		public List<ArrayList<Neuron>> getHiddenLayers(){
			return hiddenLayers;
		}
		
		public List<Neuron> getOutputNeurons(){
			return outputNeurons;
		}
	}
	
	private static NeuralNet instance = null;
	 
	public static NeuralNet getInstance(Random rand){
		if(instance == null){
			synchronized(NeuralNet.class){
				if(instance == null){
					instance = new NeuralNet(rand);
				}
			}
		}
		return instance;
	}
	
	private NeuralNet(Random rand){
		this.rand = rand;
		this.myRand = new Random();
		this.hiddenLayers = new ArrayList<ArrayList<Neuron>>();
		this.outputNeurons = new ArrayList<Neuron>();
	}
	
	private void initializeMomentum(Scanner reader){
		momentum = 0;
		System.out.println("Would you like to use Momentum: [yes/no]");
		boolean cantType = true;
		reader.nextLine();
		do{
			String input = reader.nextLine();
			if(input.equalsIgnoreCase("yes")){
				cantType = false;
				useMomentum = true;
				System.out.println("what would you like your momentum term to be?");
				momentum = reader.nextDouble();
			}
			else if(input.equalsIgnoreCase("no")){
				cantType = false;
				useMomentum = false;
			}
			else{
				System.out.println("try again");
			}
		}while(cantType);
	}
	
	private void initializeHiddenLayers(Scanner reader, int numInputs){
		boolean canMoveOn = false;
		int numHiddenLayers;
		do{
			System.out.println("How many hidden layers would you like: ");
			numHiddenLayers = reader.nextInt();
			if(numHiddenLayers > 0){
				canMoveOn = true;
			}
			else{
				System.out.println("for neural net the number of hidden layers must be > 0");
			}
		}while(!canMoveOn);
		System.out.println("the dataset has " + numInputs + " features");
		for(int i = 0; i < numHiddenLayers; i++){
			hiddenLayers.add(new ArrayList<Neuron>());
			System.out.println("How many hidden neurons would you like in layer " + i + ": ");
			int numberOfHiddenNeurons = reader.nextInt();
			for(int n = 0; n < numberOfHiddenNeurons; n++){
				if(i == 0){
					hiddenLayers.get(i).add(new HiddenNeuron(numInputs, myRand));
				}
				else{
					hiddenLayers.get(i).add(new HiddenNeuron(hiddenLayers.get(i-1).size(), myRand));
				}
				
			}
		}
	}
	
	private void initializeOutputLayer(Scanner reader){
		System.out.println("How many outputNodes?");
		int numberOfOutputNeurons = reader.nextInt();
		for(int i = 0; i < numberOfOutputNeurons; i++){
			outputNeurons.add(new OutputNeuron(hiddenLayers.get(hiddenLayers.size()-1).size(), myRand));
		}
	}
	
	private void initializeWeights(){
		for(List<Neuron> layer : hiddenLayers){
			for(Neuron neuron : layer){
				neuron.initializeIncomingWeights();
			}
		}
		for(Neuron neuron : outputNeurons){
			neuron.initializeIncomingWeights();
		}
	}
	
	private void checkTesting(Scanner reader){
		boolean cantType = true;
		do{
			System.out.println("are you testing: [yes/no]");
			String input = reader.next();
			if(input.equalsIgnoreCase("yes")){
				testing = true;
				learningRate = .175;
				momentum = .9;
				useMomentum = true;
				cantType = false;
			}
			else if(input.equalsIgnoreCase("no")){
				testing = false;
				cantType = false;
			}
		}while(cantType);
	}
	
	private void initializeNetwork(int numInputs){
		Scanner reader = new Scanner(System.in);  // Reading from System.in
		
		checkTesting(reader);
		
		if(!testing){
			System.out.println("What would you like the Learning Rate to be: ");
			learningRate = reader.nextDouble();
			
			initializeMomentum(reader);
			
			System.out.println("How many epochs without improving before the algorithm should stop? ");
			stoppingCriteria = reader.nextInt();
		}	
		initializeHiddenLayers(reader, numInputs);
		
		initializeOutputLayer(reader);
		
		initializeWeights();
	}
	
	private void updateAllWeights(double[] inputs){
		for(List<Neuron> l : hiddenLayers){
			for(Neuron n : l){
				n.updateWeights(inputs);
			}
		}
		for(Neuron n : outputNeurons){
			n.updateWeights(inputs);
		}
	}
	
	/**
	 * computes and stores the error of each neuron in that neuron
	 * @param targets
	 * @return mean squared error for that prediction
	 */
	private double computeErrors(double[] targets){ // we compute all errors before we change any weights right?
		double mse = 0; // mean squared error
		double totalSquaredError = 0;
		int numberOfErrors = 0;
		for(int n = 0; n < outputNeurons.size(); n++){
			// this will only  work for nominal datasets
//			double target;
//			if(n == targets[0]){
//				target = 1.0;
//			}
//			else{
//				target = 0;
//			}
			double error = outputNeurons.get(n).computeError(targets[0]);
			numberOfErrors++;
			totalSquaredError += Math.pow(error, 2);
			//outputNeurons.get(n).computeError(targets[0]);
		}
		for(int i = hiddenLayers.size()-1; i >= 0; i--){ // make sure error is being propagated backwards
			for(Neuron n : hiddenLayers.get(i)){
				double error = n.computeError(0.0);// the parameter is useless for hidden nodes
				numberOfErrors++;
				totalSquaredError += Math.pow(error, 2);
			}
		}
		mse = totalSquaredError / numberOfErrors;
		return mse;
	}

	public double getLearningRate(){
		return learningRate;
	}
	
	public double getMomentum(){
		return momentum;
	}
	
	public boolean useMomentum(){
		return useMomentum;
	}
	
	private void printWeights(){
		System.out.println("descending gradient...");
		if(!testing){
			return;
		}
		outputNeurons.get(0).printWeights();
		for(Neuron n : hiddenLayers.get(0)){
			n.printWeights();
		}
		System.out.println();
	}
	private void printOutputs(){
		System.out.println("forward propagating...");
		if(!testing){
			return;
		}
		outputNeurons.get(0).printOutput();
		hiddenLayers.get(0).get(0).printOutput();
		hiddenLayers.get(0).get(1).printOutput();
		hiddenLayers.get(0).get(2).printOutput();
		System.out.println();
	}
	private void printErrors(){
		System.out.println("Back Propagating...");
		if(!testing){
			return;
		}
		outputNeurons.get(0).printError();
		hiddenLayers.get(0).get(0).printError();
		hiddenLayers.get(0).get(1).printError();
		hiddenLayers.get(0).get(2).printError();
		System.out.println();
	}
	private void printPattern(double[] inputs, double[] answers){
		System.out.println("Pattern: {" + inputs[0] + ", " + inputs[1] + ", " + answers[0] + "}");
	}
	
	/**
	 * does one epoch of training
	 * @param features
	 * @param labels
	 * @return double mean squared error for that epoch
	 * @throws Exception
	 */
	private double completeEpoch(Matrix features, Matrix labels) throws Exception{
		if(!testing){
			features.shuffle(rand, labels);
		}
		double sumMSEs = 0;
		for(int r = 0; r < features.rows(); r++){
			double[] inputs = features.row(r);
			double[] answers = new double[labels.row(r).length];
			for(int i = 0; i < labels.row(r).length; i++){ // deep copy
				answers[i] = labels.row(r)[i];
			}
			printPattern(inputs, answers);
			predict(inputs, answers); // answers should probably be changed here
			printOutputs();
			sumMSEs += computeErrors(labels.row(r)); // shallow vs deep hopefully this is the original value of answers not the updated answer
			printErrors();
			updateAllWeights(inputs);
			printWeights();
		}
		return sumMSEs / features.rows();
	}
	
	/**
	 * just like complete epoch except it just computes the error for that epoch and does not update the weights
	 * @param features
	 * @param labels
	 * @return double the mean squared error for that epoch
	 * @throws Exception
	 */
	private double mse(Matrix features, Matrix labels) throws Exception{
		double sumMSEs = 0;
		for(int r = 0; r < features.rows(); r++){
			double[] inputs = features.row(r);
			double[] answers = new double[labels.row(r).length];
			for(int i = 0; i < labels.row(r).length; i++){ // deep copy
				answers[i] = labels.row(r)[i];
			}
			predict(inputs, answers); // answers should probably be changed here
			sumMSEs += computeErrors(labels.row(r)); // shallow vs deep hopefully this is the original value of answers not the updated answer
		}
		return sumMSEs / features.rows();
	}
	
	/*
	 * Create one graph with 
	 * 		the MSE (mean squared error) on the training set, 
	 * 		the MSE on the VS, 
	 * 		and the classification accuracy (% classified correctly) of the VS on the y-axis, 
	 * 		and number of epochs on the x-axis. 
	 * 		(Note two scales on the y-axis). 
	 * The results for the different measurables should be shown with a different color, line type, etc. Typical backpropagation accuracies for the 
	 * Iris data set are 85-95%.  (Showing this all in one graph is best, but if you need to use two graphs, that is OK).
	*/
	
	private void outputMSEToCSV(PrintWriter mseFile, int epochs, double mseTraining, double mseValidation){
		mseFile.print(epochs);
		mseFile.print(", ");
		mseFile.print(mseTraining);
		mseFile.print(", ");
		mseFile.print(mseValidation);
		mseFile.print("\n");
	}
	
	private void outputAccuracyToCSV(PrintWriter accuracyFile, int epochs, double accuracy){
		accuracyFile.print(epochs);
		accuracyFile.print(", ");
		accuracyFile.print(accuracy);
		accuracyFile.print("\n");
	}
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		// features.normalize(); test set is not normalized so it is unwise for me to normalize the training set
		PrintWriter mseFile = new PrintWriter("xl/MSE_vs_epochs.csv");
		PrintWriter accuracyFile = new PrintWriter("xl/classificationAccuracy.csv");		  
		double percent = .25;
//		int rowCount = (int)(features.rows() * percent);
//		Matrix validationSet = new Matrix(features, 0, 0, rowCount, features.cols()-0); //trying to ignore train vs test, and the identity of the reader so that it will generalize better
//		Matrix vsLabels = new Matrix(labels, 0, 0, rowCount, labels.cols());
//		Matrix trainingSet = new Matrix(features, rowCount, 0, features.rows()-rowCount, features.cols()-0);
//		Matrix tsLabels = new Matrix(labels, rowCount, 0, features.rows()-rowCount, labels.cols());
		initializeNetwork(features.cols());
		double mseValidation, mseTraining, accuracy;
		BestSolutionSoFar bssf = new BestSolutionSoFar();
		int epochs =  0;
		do{
			System.out.println("---Epoch " + (epochs+1) + "---");
			mseTraining = completeEpoch(features, labels);
			epochs++;
			
			//mseValidation = mse(validationSet, vsLabels);
			//outputMSEToCSV(mseFile, epochs, mseTraining, mseValidation);
			
			//Matrix confusion = new Matrix();
			//accuracy = measureAccuracy(validationSet, vsLabels, confusion);
			//outputAccuracyToCSV(accuracyFile, epochs, accuracy);
			if(testing && epochs == 3){
				break;
			}
		}while(true /*bssf.hasImprovedOverLastNEpochs(mseValidation, mseTraining, accuracy)*/);
		accuracyFile.close();
		mseFile.close();
		//hiddenLayers = bssf.getHiddenLayers();
		//outputNeurons = bssf.getOutputNeurons();
		System.out.println("epochs: " + epochs);
		System.out.println("mseTrS: " + bssf.getMseTraining());
		System.out.println("mseVS: " + bssf.getMseValidation());
		System.out.println("VS accuracy: " + bssf.getAccuracy());
	}
	
	@Override
	public void mseTest(Matrix testFeatures, Matrix testLabels) {
//		double mseTest = -1;
//		double testAccuracy = -1;
//		Matrix streamlinedTestFeatures = new Matrix(testFeatures, 0, 2, testFeatures.rows(), testFeatures.cols()-2);
//		try {
//			mseTest = mse(streamlinedTestFeatures, testLabels);
//		} catch (Exception e) {
//			e.printStackTrace();
//		}
//		Matrix confusion = new Matrix();
//		try {
//			testAccuracy = measureAccuracy(streamlinedTestFeatures, testLabels, confusion);
//		} catch (Exception e) {
//			e.printStackTrace();
//		}
//		System.out.println("mseTeS: " + mseTest);
//		System.out.println("test accuracy: " + testAccuracy);
	}
	
	private double[] getOutputs(List<Neuron> layer, double[] inputs){
		double[] outputs = new double[layer.size()];
		for(int i = 0; i < layer.size(); i++){
			outputs[i] = layer.get(i).activate(inputs);
		}
		return outputs;
	}

	private double singleOutputFromMultipleOutputNodes(double[] answers){
		double greatest = 0;
		int indexOfGreatest = -1;
		for(int i = 0; i < answers.length; i++){
			if(answers[i] > greatest){
				greatest = answers[i];
				indexOfGreatest = i;
			}
		}
		if(indexOfGreatest == -1){
			System.out.println("something screwy happened in the predict funciton");
			System.exit(0);
		}
		return (double)indexOfGreatest;
	}
	
	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		double[] layerOutputs = null;
		for(int i = 0; i < hiddenLayers.size(); i++){
			if(i == 0){
				layerOutputs = getOutputs(hiddenLayers.get(i), features);
			}
			else{
				layerOutputs = getOutputs(hiddenLayers.get(i), layerOutputs);
			}
		}
		double[] answers = getOutputs(outputNeurons, layerOutputs); 
		if(answers.length > 1){
			labels[0] = singleOutputFromMultipleOutputNodes(answers);
		}
		else{
			labels[0] = answers[0];
		}
	}
}