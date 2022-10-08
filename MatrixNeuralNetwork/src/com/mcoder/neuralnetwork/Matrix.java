package com.mcoder.neuralnetwork;

import java.io.Serializable;
import java.util.function.Function;

public class Matrix implements Serializable {
	private static final long serialVersionUID = 1L;
	
	private double[][] data;
	private int rows, cols;
	
	public Matrix(int rows, int cols) {
		this.rows = rows;
		this.cols = cols;
		data = new double[rows][cols];
	}
	
	/**
	 * It sums the given matrix to this one
	 * @param m Matrix to sum
	 */
	public void add(Matrix m) {
		if (rows == m.rows && cols == m.cols)
			map((d, i, j) -> d+m.data[i][j]);
	}
	
	/**
	 * It makes an element-by-element multiplication
	 * between this matrix and the given one
	 * @param m Matrix to multiply
	 */
	public void multiply(Matrix m) {
		if (rows == m.rows && cols == m.cols)
			map((d, i, j) -> d*m.data[i][j]);
	}
	
	/**
	 * It multiplies this matrix to the given value
	 * @param v
	 */
	public void multiply(double v) {
		map((d, i, j) -> d*v);
	}
	
	/**
	 * It maps the given function on the current matrix
	 * and updates matrix data with returned values
	 * @param func Function to use to do the mapping
	 */
	public void map(MatrixFunction<Double, Double> func) {
		Matrix r = map(this, func);
		data = r.data;
	}
	
	public void map(Function<Double, Double> func) {
		Matrix r = map(this, func);
		data = r.data;
	}
	
	/**
	 * It makes a matrix transposition
	 */
	public void transpose() {
		Matrix r = transpose(this);
		data = r.data;
		rows = r.rows;
		cols = r.cols;
	}
	
	public double[] toArray() {
		if (cols == 1) {
			double[] array = new double[rows];
			for (int i = 0; i < rows; i++)
				array[i] = data[i][0];
			return array;
		}
		
		return null;
	}
	
	/**
	 * It makes a subtraction between to matrices
	 * @param m1
	 * @param m2
	 * @return The matrix containing resulting values
	 */
	public static Matrix sub(Matrix m1, Matrix m2) {
		if (m1.rows == m2.rows && m1.cols == m2.cols)
			return map(m1, (d, i, j) -> d-m2.data[i][j]);
		return null;
	}
	
	/**
	 * It makes the mapping of the given function on each element of the matrix m
	 * @param m Matrix to map on
	 * @param func Function to use to do the mapping
	 * @return
	 */
	public static Matrix map(Matrix m, MatrixFunction<Double, Double> func) {
		Matrix r = new Matrix(m.rows, m.cols);
		for (int i = 0; i < m.rows; i++)
			for (int j = 0; j < m.cols; j++)
				r.data[i][j] = func.apply(m.data[i][j], i, j);
		return r;
	}
	
	public static Matrix map(Matrix m, Function<Double, Double> func) {
		return map(m, (d, i, j) -> func.apply(d));
	}
	
	/**
	 * It makes a matrix multiplication between two matrices
	 * @param m1
	 * @param m2
	 * @return The matrix which contains the resulting values
	 */
	public static Matrix matrixMultiply(Matrix m1, Matrix m2) {
		if (m1.cols != m2.rows)
			return null;
		
		Matrix r = new Matrix(m1.rows, m2.cols);
		for (int i = 0; i < m1.rows; i++)
			for (int j = 0; j < m2.cols; j++) {
				double sum = 0;
				for (int k = 0; k < m1.cols; k++)
					sum += m1.data[i][k]*m2.data[k][j];
				r.data[i][j] = sum;
			}
		
		return r;
	}
	
	/**
	 * It transpose the given matrix
	 * @param m Matrix to transpose
	 * @return Matrix which contains the resulting values
	 */
	public static Matrix transpose(Matrix m) {
		Matrix r = new Matrix(m.cols, m.rows);
		r.map((d, i, j) -> m.data[j][i]);
		return r;
	}
	
	/**
	 * Create a single-column Matrix object starting with an array of doubles
	 * @param array The array to use to create the matrix
	 * @return The resulting matrix
	 */
	public static Matrix fromArray(double[] array) {
		Matrix m = new Matrix(array.length, 1);
		for (int i = 0; i < array.length; i++)
			m.data[i][0] = array[i];
		return m;
	}
}
