import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;


public class DecisionTree extends SupervisedLearner {
	Tree<Matrix> tree;
	private double bestAccuracy;
	boolean earlyStoppingCriteria = true;
	
	private double logBase2(double value){
		return Math.log(value)/Math.log(2);
	}
	
	/**
	 * calculates and returns the information for a specific value of an attribute
	 * @param treeNode the current node of the tree
	 * @param col the column or attribute that info is being gathered for
	 * @param value the value of the given attribute/column
	 * @return information for the provided value of the provided column/attribute
	 */
	private double infoForValue(Tree<Matrix>.Node<Matrix> treeNode, int col, double value){
		double infoForValue = 0;
		Matrix features = treeNode.data;
		Matrix labels = treeNode.data2;
		MatrixParser mParser = new MatrixParser(features, labels);
		double ratio = mParser.getRatio(col, value);
		double innerSum = 0;
		for(int cv = 0; cv < labels.valueCount(0); cv++){
			double proportion = mParser.getProportion(col, value, cv);
			if(proportion > 0) innerSum -= proportion * logBase2(proportion);
		}
		return ratio*innerSum;
	}
	
	/**
	 * calculates the information for the various columns/attributes in treeNode
	 * @param treeNode the current node of the tree
	 * @param c the column/attribute for which information is being calculated
	 * @return the information that is left should the tree split on column/attribute c
	 */
	private double calculateInfo(Tree<Matrix>.Node<Matrix> treeNode, int c){
		MatrixParser mParser = new MatrixParser(treeNode.data, treeNode.data2);
		double info = 0;
		double v;
		for(v = 0; v < treeNode.data.valueCount(c); v++){
			info += infoForValue(treeNode, c, v);
		}
		//for unknown values
		v = Double.MAX_VALUE;
		info += infoForValue(treeNode, c, v);
		return info;
	}
	
	/**
	 * creates a branch for the value of the attribute that is being split on in the current treeNode 
	 * @param treeNode the current node of the tree
	 * @param v the value of the attribute that the branch is being created from
	 * @param info the amount of information that is left for the subtree where treeNode is the root 
	 * @throws Exception
	 */
	private void createBranch(Tree<Matrix>.Node<Matrix> treeNode, double v, double info) throws Exception{
		Matrix features = treeNode.data;
		Matrix labels = treeNode.data2;
		int attributeToSplitOn = treeNode.getAttributeToSplitOn();
		MatrixParser mParser = new MatrixParser(features, labels);
		Matrix[] subMatrices = mParser.splitFeaturesAndLabels(attributeToSplitOn, v);
		Matrix subFeatures = subMatrices[0];
		Matrix subLabels = subMatrices[1];
		treeNode.addChild(subFeatures, subLabels);
		
		double majorityClass;
		if(subFeatures != null) majorityClass = subLabels.mostCommonValue(0);
		else majorityClass = labels.mostCommonValue(0);
		try{
			treeNode.getChild((int)v).setLabel(majorityClass);
		}catch(IndexOutOfBoundsException e){
			treeNode.getChild(treeNode.numChildren()-1).setLabel(majorityClass);
		}
				
		if(subFeatures != null){
			try{
				rSplitOnAttributes(treeNode.getChild((int)v), info);
			}catch(IndexOutOfBoundsException e){ // because if the child is unknown the value of idx will be Double.MAX_VALUE but it will just be the last child added
				rSplitOnAttributes(treeNode.getChild(treeNode.numChildren()-1), info);
    		}
		}
	}
	
	private void rSplitOnAttributes(Tree<Matrix>.Node<Matrix> treeNode, double infoAfterLastSplit) throws Exception{
		Matrix features = treeNode.data;
		int depth = treeNode.getDepth();
		if(infoAfterLastSplit == 0 || depth+1 == features.cols()){
			return;
		}
		
		double minInfo = Double.MAX_VALUE;
		int attributeToSplitOn = -1;
		for(int c = 0; c < features.cols(); c++){
			double info = calculateInfo(treeNode, c);
			if(info < minInfo){
				attributeToSplitOn = c;
				minInfo = info;
			}
		}
		
		if(earlyStoppingCriteria){
			if(minInfo < infoAfterLastSplit){
				treeNode.setAttributeToSplitOn(attributeToSplitOn);
				for(int v = 0; v < features.valueCount(attributeToSplitOn); v++){
					createBranch(treeNode, v, minInfo);
				}
				//branch for unknown
				double v = Double.MAX_VALUE;
				createBranch(treeNode, v, minInfo);
			}
		}
		else{
			treeNode.setAttributeToSplitOn(attributeToSplitOn);
			for(int v = 0; v < features.valueCount(attributeToSplitOn); v++){
				createBranch(treeNode, v, minInfo);
			}
			//branch for unknown
			double v = Double.MAX_VALUE;
			createBranch(treeNode, v, minInfo);
		}
		
	}
	
	private boolean rPrune(Tree<Matrix>.Node<Matrix> node, Matrix vsFeatures, Matrix vsLabels) throws Exception{
		int actualAttributeToSplitOn = node.getAttributeToSplitOn();
		node.setAttributeToSplitOn(-1);
		Matrix confusion = new Matrix();
		double accuracy = measureAccuracy(vsFeatures, vsLabels, confusion);
		if(accuracy >= bestAccuracy){
			System.out.println("tree nodes: " + tree.getNumNodes());
			bestAccuracy = accuracy;
			System.out.println("accuracy: " + accuracy);
			for(int i = node.children.size()-1; i > -1; i--){
				node.children.remove(node.children.get(i));
			}
			//tree.removeNode(node);
			System.out.println("tree nodes: " + tree.getNumNodes());
			accuracy = measureAccuracy(vsFeatures, vsLabels, confusion);
			System.out.println("accuracy: " + accuracy); 
			return true;
		}
		else{
			node.setAttributeToSplitOn(actualAttributeToSplitOn);
			for(int i = 0; i < node.children.size(); i++){
				Tree<Matrix>.Node<Matrix> child = node.getChild(i);
				rPrune(child, vsFeatures, vsLabels);				
			}
			return false;
		}
	}
	
	private void pruneTree(Matrix vsFeatures, Matrix vsLabels) throws Exception{
		Matrix confusion = new Matrix();
		bestAccuracy = measureAccuracy(vsFeatures, vsLabels, confusion);
		Tree<Matrix>.Node<Matrix> curr = tree.getRoot();
		rPrune(curr, vsFeatures, vsLabels);
	}
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		features.shuffle(new Random(), labels);
		Matrix trainingFeatures, trainingLabels, vsFeatures, vsLabels;
		double percentValidation = .2;
		trainingFeatures = new Matrix(features, 0, 0, (int)(features.rows()*(1-percentValidation)), features.cols());
		trainingLabels = new Matrix(labels, 0, 0, (int)(labels.rows()*(1-percentValidation)), labels.cols());
		vsFeatures = new Matrix(features, (int)(features.rows()*(1-percentValidation)), 0, (int)(features.rows()*(percentValidation)), features.cols());
		vsLabels = new Matrix(labels, (int)(labels.rows()*(1-percentValidation)), 0, (int)(labels.rows()*(percentValidation)), labels.cols());
		
		tree = new Tree<Matrix>(trainingFeatures, trainingLabels);
		double majorityClass = labels.mostCommonValue(0);
		tree.getRoot().setLabel(majorityClass);
		rSplitOnAttributes(tree.getRoot(), Double.MAX_VALUE);
		
		if(!earlyStoppingCriteria)pruneTree(vsFeatures, vsLabels);
		
		System.out.println("tree depth: " + tree.getDepth());
		System.out.println("tree nodes: " + tree.getNumNodes());
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		Tree<Matrix>.Node<Matrix> curr = tree.getRoot();
		double answer = curr.getLabel();
		while(curr.getAttributeToSplitOn() != -1){
			int attr  = curr.getAttributeToSplitOn();
			try{
				curr = curr.getChild((int)features[attr]);
			}catch(IndexOutOfBoundsException e){
				curr = curr.getChild(curr.numChildren()-1);
			}
			answer = curr.getLabel();
		}
		labels[0] = answer;
	}

	@Override
	public void mseTest(Matrix features, Matrix labels) {
	}

}
