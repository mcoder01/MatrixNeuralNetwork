package com.mcoder.neuralnetwork;

public interface MatrixFunction<T, R> {
	/**
	 * Function used to execute instructions 
	 * for each element of a Matrix
	 * @param data Current value of the matrix
	 * @param row Corresponding row
	 * @param col Corresponding column
	 */
	public R apply(T data, int row, int col);
}
