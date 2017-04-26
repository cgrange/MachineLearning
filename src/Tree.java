import java.util.ArrayList;
import java.util.List;

public class Tree<T> {
    private Node<T> root;
    private int numNodes;
    private int treeDepth;
    
    public Tree(T rootData, T rootData2) {
        root = new RootNode<T>(rootData, rootData2);
    }
    
    public abstract class Node<T>{
    	protected T data;
    	protected T data2;
        protected List<Node<T>> children;  
        private int attributeToSplitOn = -1;
        private double label = -1;
        
    	public void addChild(T data, T data2){
        	Node<T> child = new ChildNode<T>(data, data2, this);
        	children.add(child);
        }
    	
    	public abstract Node<T> getParent();
    	
    	public Node<T> getChild(int idx){
    		return children.get(idx);
    	}
    	
    	public void setLabel(double majorityClass){
    		this.label = majorityClass;
    	}
    	
    	public double getLabel(){
    		return label;
    	}
    	
    	public void setAttributeToSplitOn(int attr){
    		this.attributeToSplitOn = attr;
    	}
    	
    	public int getAttributeToSplitOn(){
    		return attributeToSplitOn;
    	}
    	
    	/**
    	 * @return depth of this node in the tree (0 based)
    	 */
    	public int getDepth(){
			Node<T> parent = getParent();
			int depth = 0;
			while(parent != null){
				depth++;
				parent = parent.getParent();
			}
			return depth;	
    	}
    	
    	public int numChildren(){
    		return children.size();
    	}
    	
    }

    public class ChildNode<T> extends Node<T>{
        private Node<T> parent; 
        
        public ChildNode(T data, T data2, Node<T> parent){
        	this.data = data;
        	this.data2 = data2;
        	this.parent = parent;
        	children = new ArrayList<Node<T>>();
        }

		@Override
		public Node<T> getParent() {
			return parent;
		}
    }
    
    public class RootNode<T> extends Node<T>{
        public RootNode(T data, T data2){
        	this.data = data;
        	this.data2 = data2;
        	children = new ArrayList<Node<T>>();
        }

		@Override
		public Node<T> getParent() {
			return null;
		}
    }
    
    public Node<T> getRoot(){
    	return root;
    }
    
    private void rRemoveNode(Node<T> node, Node<T> target){
    	if(node.children.contains(target)){
    		node.children.remove(target);
    	}
    	else{
    		for(Node<T> child : node.children){
    			rRemoveNode(child, target);
    		}
    	}
    }

	public void removeNode(Node<T> node) {
		rRemoveNode(root, node);
	}
    
	private void rGetNumNodes(Node<T> curr){
		numNodes++;
		for(Node<T> child : curr.children){
			rGetNumNodes(child);
		}
	}
	
	public int getNumNodes(){
		numNodes = 0;
		Node<T> curr = root;
		rGetNumNodes(curr);
		return numNodes;
	}
	
	private void rGetDepth(Node<T> curr){
		int currDepth = curr.getDepth();
		if(currDepth > treeDepth){
			treeDepth = currDepth;
		}
		for(Node<T> child : curr.children){
			rGetDepth(child);
		}
	}
	
	public int getDepth(){
		treeDepth = 0;
		Node<T> curr = root;
		rGetDepth(curr);
		return treeDepth;		
	}
}
