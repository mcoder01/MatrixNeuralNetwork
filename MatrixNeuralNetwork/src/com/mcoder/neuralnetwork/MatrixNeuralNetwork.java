package com.mcoder.neuralnetwork;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

public class MatrixNeuralNetwork implements Serializable {
	private static final long serialVersionUID = 1L;
	
	private Matrix[] weights, biases, outputs;
	private int layers;
	private double lr;
	
	/**
	 * Construct the neural network starting with an array
	 * containing the number of nodes for each level
	 * @param nodes Array containing number of nodes for each level of the network
	 */
	public MatrixNeuralNetwork(int... nodes) {
		layers = nodes.length;
		weights = new Matrix[layers-1];
		biases = new Matrix[layers-1];
		outputs = new Matrix[layers];
		
		// Initial weights of the network are randomized
		for (int i = 0; i < weights.length; i++) {
			weights[i] = new Matrix(nodes[i], nodes[i+1]);
			randomize(weights[i]);
			
			biases[i] = new Matrix(nodes[i+1], 1);
			randomize(biases[i]);
		}
		
		// Default learning ratio
		lr = 1;
	}
	
	/**
	 * It executes a randomization of the given matrix.
	 * Used to randomize weights and biases on initialization
	 * @param m Matrix to randomize
	 */
	private void randomize(Matrix m) {
		m.map((d, i, j) -> Math.random()*2-1);
	}
	
	/**
	 * It executes the feed-forward operation on the network
	 * based on the given inputs
	 * @param inputValues Values on which to start the prediction
	 * @return Array of doubles containing the predicted outputs
	 */
	public double[] predict(double[] inputValues) {
		// The output values of the first layer of the network corresponds to input values
		outputs[0] = Matrix.fromArray(inputValues);
		for (int i = 1; i < layers; i++) {
			Matrix transposedOutput = Matrix.transpose(outputs[i-1]);
			outputs[i] = Matrix.matrixMultiply(transposedOutput, weights[i-1]);
			outputs[i].transpose();
			outputs[i].add(biases[i-1]);
			outputs[i].map(this::sigmoid); // Nodes activation
		}
		
		return outputs[layers-1].toArray();
	}
	
	/**
	 * This function executes the training of the network
	 * using target outputs to calculate the gradients
	 * and to adjust weights and biases depending on 
	 * the calculated errors
	 * @param inputs Array of double containing the input values
	 * @param targetArray Array of double containing the expected values
	 */
	public void train(double[] inputs, double[] targetArray) {
		predict(inputs); // First of all, we execute a prediction to calculate some outputs
		Matrix targets = Matrix.fromArray(targetArray);
		Matrix errors = Matrix.sub(targets, outputs[layers-1]);
		for (int i = layers-1; i > 0; i--) {
			// Calculate gradients basing on the calculated outputs and using
			// the derivative of the sigmoid function
			Matrix gradients = Matrix.map(outputs[i], this::dsigmoid);
			gradients.multiply(errors);
			gradients.multiply(lr);
			biases[i-1].add(gradients); // Adjusts the biases
			
			gradients.transpose();
			Matrix deltaWeights = Matrix.matrixMultiply(outputs[i-1], gradients);
			weights[i-1].add(deltaWeights); // Adjusts the weights
			
			errors = Matrix.matrixMultiply(weights[i-1], errors);
		}
	}
	
	/**
	 * It saves the neural network on the file system
	 * @param path The file path to save the network to
	 * @throws IOException When the I/O operation fails
	 */
	public void save(String path) throws IOException {
		FileOutputStream fos = new FileOutputStream(path);
		ObjectOutputStream out = new ObjectOutputStream(fos);
		out.writeObject(this);
		out.close();
	}
	
	/**
	 * It loads the network from the file system
	 * @param path The file path which contains the network
	 * @return The MatrixNeuralNetwork loaded object
	 * @throws IOException When the I/O operation fails
	 * @throws ClassNotFoundException When the corresponding class of the loaded object can't be found
	 */
	public static MatrixNeuralNetwork load(String path) throws IOException, ClassNotFoundException {
		FileInputStream fis = new FileInputStream(path);
		ObjectInputStream in = new ObjectInputStream(fis);
		MatrixNeuralNetwork mnn = (MatrixNeuralNetwork) in.readObject();
		in.close();
		return mnn;
	}
	
	/**
	 * Activation function. It maps the calculated output of a neuron
	 * in a range from -1 to 1
	 * @param x The value to map
	 * @return The mapped value
	 */
	private double sigmoid(double x) {
		return 1.0/(1.0+Math.exp(-x));
	}
	
	/**
	 * Derivative of the activation function. Used to calculate the gradients
	 * during the training of the network
	 * @param y The value to map
	 * @return The mapped value
	 */
	private double dsigmoid(double y) {
		return y*(1.0-y);
	}
	
	public void setLR(double lr) {
		this.lr = lr;
	}
}
