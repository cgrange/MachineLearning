import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;


public class Kmeans extends SupervisedLearner{
	Matrix instances;
	List<double[]> twoCentroidsAgo;
	List<double[]> lastCentroids;
	List<double[]> centroids;
	boolean centroidsDidntChange;
	
	public Kmeans(){
		centroids = new ArrayList<double[]>();
	}
	
	private boolean areEqual(List<double[]> list1, List<double[]> list2){
		for(int i = 0; i < list1.size(); i++){
			double[] a1, a2;
			try{
				a1 = list1.get(i);
				a2 = list2.get(i);
			}catch(Exception e){
				return false;
			}
			if(!Arrays.equals(a1, a2)){
				return false;
			}
		}
		return true;
	}
	
	private List<Set<Integer>> rSeparateIntoClusters(int k, List<Set<Integer>> clusters){
		if(centroidsDidntChange){
			return clusters;
		}
		centroidsDidntChange = true;
		clusters = separateInstances(k);
		recalculateCentroids(clusters);
		if(areEqual(centroids, twoCentroidsAgo)){
			return clusters;
		}
		return rSeparateIntoClusters(k, clusters);
	}
	
	private double getDistance(double[] centroid, double[] instance){
		
		double distanceSquared = 0;
		
		for(int c = 0; c< centroid.length; c++){
			if(instance[c] == Double.MAX_VALUE || centroid[c] == Double.MAX_VALUE){ // the value is unknown
				distanceSquared += 1; //maximum distance in a normalized set
			}
			else if(this.instances.valueCount(c) != 0){// the value is  nominal
				if(c == 8){//just for abalone dataset so that rings are treated as continuous even though they are nominal
					distanceSquared += Math.pow((centroid[c] - instance[c]), 2);
				}
				if(centroid[c] != instance[c]){
					distanceSquared += 1; //maximum distance in a normalized set (I'm assuming that different classes are entirely unique)
				}
				else distanceSquared += 0;
			}
			else{// the value is continuous
				distanceSquared += Math.pow((centroid[c] - instance[c]), 2);
			}
		}
		double distance = Math.sqrt(distanceSquared);
		return distance;
	}
	
	private void outputAssignments(List<Set<Integer>> clusters){
		for(int i = 0; i < instances.rows(); i++){
			if(i%10 == 0){
//				System.out.println();
			}
			for(int c = 0; c < clusters.size(); c++){
				if(clusters.get(c).contains(i)){
//					System.out.print(i + "=" + c + ", ");
					break;
				}
			}
		}
//		System.out.println();
	}
	
	/**
	 * separates all the instances in the training set into k clusters 
	 * @param k
	 * @return a list of length k which holds clusters (sets) which hold the indexes of the instances that are in that cluster 
	 */
	private List<Set<Integer>> separateInstances(int k){
		List<Set<Integer>> clusters = new ArrayList<Set<Integer>>();
		for(int i = 0; i < k; i++){
			clusters.add(new HashSet<Integer>());
		}
		for(int r = 0; r < instances.rows(); r++){
			double minDistance = Double.MAX_VALUE;
			int cluster = -1;
			for(int i = 0; i < k; i++){
				double distance = getDistance(centroids.get(i), instances.row(r));
				if(distance < minDistance){
					minDistance = distance;
					cluster = i;
				}
			}
			clusters.get(cluster).add(r);
		}
		outputAssignments(clusters);
		return clusters;
	}
	
	/**
	 * goes through all the instances in a cluster and returns the most common value in column c for all of those instances
	 * @param cluster 
	 * @param c the column to look at in each instance
	 * @return most common nominal value that isn't unknown, unless all of them are unknown, for this cluster
	 */
	private double mcv(Set<Integer> cluster, int c){
		double mcv = Double.MAX_VALUE;
		Map<Double, Integer> valueToInstancesMap = new HashMap<Double, Integer>();
		Iterator<Integer> it = cluster.iterator();
		while(it.hasNext()){
			int index = it.next();
			double val = instances.get(index, c);
			if(valueToInstancesMap.containsKey(val)){
				int instances = valueToInstancesMap.get(val);
				valueToInstancesMap.put(val, instances+1);
			}
			else{
				valueToInstancesMap.put(val, 1);
			}
		}
		Set<Double> keySet = valueToInstancesMap.keySet();
		Iterator<Double> it2 = keySet.iterator();
		int maxInstances = 0;
		while(it2.hasNext()){
			double key = it2.next();
			int instances = valueToInstancesMap.get(key);
			if(instances > maxInstances){
				if(key != Double.MAX_VALUE){
					maxInstances = instances;
					mcv = key;
				}
			}
			else if(instances == maxInstances){
				if(key < mcv){
					mcv = key;
				}
			}
		}
		return mcv;
	}
	
	/**
	 * goes through all the instances in a cluster and returns the mean in column c for all of those instances
	 * @param cluster 
	 * @param c the column to look at in each instance
	 * @return the average value of column c for all the instances in the cluster
	 */
	private double mean(Set<Integer> cluster, int c){
		double sum = 0;
		Iterator<Integer> it = cluster.iterator();
		double denominator = cluster.size();
		while(it.hasNext()){
			int index = it.next();
			double val = instances.get(index, c);
			if(val != Double.MAX_VALUE){
				sum += instances.get(index, c);
			}
			else{
				denominator -= 1;
			}
		}
		if(sum == 0){
			return Double.MAX_VALUE;
		}
		double mean = sum/denominator;
		return mean;
	}
	
	/**
	 * deep copies the values of list1 into list2 so that list2 has the same values as list1 but not the same reference
	 * @param list1
	 * @param list2
	 */
	private void deepCopy(List<double[]> list1, List<double[]> list2){
		if(list1 == null){
			return;
		}
		if(list2 != null) list2.clear();
		else list2 = new ArrayList<double[]>();
		for(int i = 0; i < list1.size(); i++){
			double[] a1 = list1.get(i);
			double[] a2 = new double[a1.length];
			for(int n = 0; n < a1.length; n++){
				a2[n] = a1[n];
			}
			list2.add(a2);
		} 
	}
	
	/**
	 * recalculates the centroids based off of the newly separated clusters
	 * @param clusters
	 */
	private void recalculateCentroids(List<Set<Integer>> clusters){
		//deep copy centroids to lastCentroids and lastCentroids to twoCentroidsAgo
		deepCopy(lastCentroids, twoCentroidsAgo);
		deepCopy(centroids, lastCentroids);
//		System.out.println("computing centroids:");
		for(int i = 0; i < clusters.size(); i++){
			Set<Integer> clusterI = clusters.get(i);
			
			double[] centroidI = new double[instances.cols()];
			
			for(int c = 0; c < instances.cols(); c++){
				if(instances.valueCount(c) > 0){ // attribute c is nominal
					centroidI[c] = mcv(clusterI, c);
				}
				else{ // attribute c is continuous
					centroidI[c] = mean(clusterI, c);
				}	
			}
//			System.out.println("centroid " + i + ": " + Arrays.toString(centroidI));
			if(!Arrays.equals(centroids.get(i), centroidI)){
				centroids.set(i, centroidI);
				centroidsDidntChange = false;
			}			
		}
	}
	
	private double sse(Set<Integer> cluster, double[] centroid){
		double sse = 0;
		Iterator<Integer> it = cluster.iterator();
		while(it.hasNext()){
			int idx = it.next();
			double distance = getDistance(centroid, instances.row(idx));
			sse += Math.pow(distance, 2);
		}
		return sse;
	}
	
	private double totalSSE(List<Set<Integer>> clusters) {
		double totalSSE = 0;
		for(int i = 0; i < clusters.size(); i++){
			double sse = sse(clusters.get(i), centroids.get(i));
			totalSSE += sse;
		}
		return totalSSE;
	}

	private void outputInfo(List<Set<Integer>> clusters) {
		System.out.println();
		System.out.println("******************** " + clusters.size() + " CLUSTERS ********************");
		System.out.println();
		
		for(int i = 0; i < centroids.size(); i++){
//			System.out.println("centroid for cluster" + i + ": " + Arrays.toString(centroids.get(i)));
//			System.out.println("cluster size: " + clusters.get(i).size());
//			double sse = sse(clusters.get(i), centroids.get(i));
//			System.out.println("sse for cluster" + i + ": " + sse);
		}
		System.out.println("validation score: " + getValidationScore(clusters));
//		double totalSSE = totalSSE(clusters);
//		System.out.println("total sse: " + totalSSE);
	}
	
	private double distanceToNearestCentroid(double[] centroid){
		double minDist = Double.MAX_VALUE;
		for(int i = 0; i < centroids.size(); i++){
			double distance = getDistance(centroid, centroids.get(i));
			if(distance < minDist && distance != 0){
				minDist = distance;
			}
		}
		return minDist;
	}
	
	private double getValidationScore(Set<Integer> cluster, int i){
		//validation score for a cluster: mse / squared distance from its centroid to closest cluster centroid
		double mse = sse(cluster, centroids.get(i))/cluster.size();
		double clusterSeparation = distanceToNearestCentroid(centroids.get(i));
		return mse/Math.pow(clusterSeparation, 2);
	}
	
	private double getValidationScore(List<Set<Integer>> clusters){
		double totalValidationScore = 0;
		for(int i = 0; i < clusters.size(); i++){
			Set<Integer> cluster = clusters.get(i);
			totalValidationScore += getValidationScore(cluster, i);
		}
		return totalValidationScore/clusters.size();
	}
	
	private double getAi(List<Set<Integer>> clusters, int instanceIdx){
		double ai = 0;
		for(Set<Integer> cluster : clusters){
			if(cluster.contains(instanceIdx)){
				double sumWithin = 0;
				for(int i : cluster){
					sumWithin += getDistance(instances.row(i), instances.row(instanceIdx));
				}
				ai = sumWithin/cluster.size();
			}
		}
		return ai;
	}
	
	private double getBi(List<Set<Integer>> clusters, int instanceIdx){
		double minDist = Double.MAX_VALUE;
		int minDistCluster = -1;
		int sameCluster = -1;
		for(int i = 0; i < clusters.size(); i++){
			Set<Integer> cluster = clusters.get(i);
			if(cluster.contains(instanceIdx)){
				sameCluster = i;
			}
		}
		for(int i = 0; i < centroids.size(); i++){
			double[] centroid = centroids.get(i);
			double dist = getDistance(centroid, instances.row(instanceIdx));
			if(dist < minDist && i != sameCluster){
				minDist = dist;
				minDistCluster = i;
			}
		}
		double sumBetween = 0;
		for(int n : clusters.get(minDistCluster)){
			sumBetween += getDistance(instances.row(instanceIdx), instances.row(n));
		}
		double bi = sumBetween/clusters.get(minDistCluster).size();
		return bi;
	}
	
	private double getSilhouetteScore(List<Set<Integer>> clusters, int instanceIdx){
		double ai = getAi(clusters, instanceIdx);
		double bi = getBi(clusters, instanceIdx);
		if(ai == 0){
			return 0;
		}
		else return 1-(ai/bi);
	}
	
	private double getSilhouetteScore(List<Set<Integer>> clusters){
		double sumSilhouette = 0;
		for(int i = 0; i < instances.rows(); i++){
			sumSilhouette += getSilhouetteScore(clusters, i);
		}
		return sumSilhouette/instances.rows();
	}
	
	private double getSilhouetteScore(Set<Integer> cluster, List<Set<Integer>> clusters){
		Iterator<Integer> it = cluster.iterator();
		double clusterSum = 0;
		while(it.hasNext()){
			int idx = it.next();
			clusterSum += getSilhouetteScore(clusters, idx);
		}
		return clusterSum/cluster.size();
	}
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception { // labels should be null for Kmeans
		instances = features;
		
		lastCentroids = new ArrayList<double[]>();
		twoCentroidsAgo = new ArrayList<double[]>();
		
		for(int k = 2; k < 8; k++){
			instances.shuffle(new Random());
//			int k = 4;
		
			centroids.clear();
			centroidsDidntChange = false;
			
//			System.out.println("computing centroids:");
			for(int i = 0; i < k; i++){ // assign k centroids to be the first k instances in the training set
//				System.out.println("centroid " + i + ": " + Arrays.toString(instances.row(i)));
				
				centroids.add(instances.row(i));	
			}
			
			List<Set<Integer>> clusters = null;
			clusters = rSeparateIntoClusters(k, clusters);
			
			outputInfo(clusters);
		}	 
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void mseTest(Matrix features, Matrix labels) {
		// TODO Auto-generated method stub
		
	}
}
