import java.util.List;


public class MatrixParser {
	Matrix features;
	Matrix labels;
	
	public MatrixParser(Matrix features, Matrix labels){
		this.features = features;
		this.labels = labels;
	}
	
	/**
	 * @param col the column (attribute) that is being tested
	 * @param value the enumerated value for the attribute value
	 * @return a double representing the ratio of the value to the attribute as a whole
	 */
	public double getRatio(int col, double attrValue){
		double attrValueInstances = 0;
		double instances = features.rows();
		for(int r = 0; r < features.rows(); r++){
			if(features.get(r, col) == attrValue){
				attrValueInstances++;
			}		
		}
		return attrValueInstances/instances;
	}
	
	/**
	 * 
	 * @param col the column (attribute) that is being tested
	 * @param attrValue the enumerated value for the attribute value
	 * @param classValue the enumerated value which represents a value of the class or the target
	 * @return proportion of the specified class value with respect to the given attribute value
	 */
	public double getProportion(int col, double attrValue, int classValue){
		double attrValueInstances = 0;
		double classValueInstances = 0;
		for(int r = 0; r < features.rows(); r++){
			if(features.get(r, col) == attrValue){
				attrValueInstances++;
				if(labels.get(r, 0) == classValue){
					classValueInstances++;
				}
			}		
		}
		return classValueInstances/attrValueInstances;
	}
	
	/**
	 * 
	 * @param col the column/attribute to split on
	 * @param v the attributeValue that the new matrix will consist of
	 * @return the sub features matrix comprised of only instances with the specified attribute value
	 * @throws Exception 
	 */
	public Matrix[] splitFeaturesAndLabels(int col, double v) throws Exception{
		Matrix[] subMatrices = new Matrix[2];
		subMatrices[0] = null;
		subMatrices[1] = null;
		for(int r = 0; r < features.rows(); r++){
			if(features.get(r, col) == v){
				if(subMatrices[0] == null){
					subMatrices[0] = new Matrix(features, r, 0, 1, features.cols());
					subMatrices[1] = new Matrix(labels, r, 0, 1, 1);
				}
				else{
					subMatrices[0].add(features, r, 0, 1);
					subMatrices[1].add(labels, r, 0, 1);
				}
			}
		}
		return subMatrices;
	}
	
}
