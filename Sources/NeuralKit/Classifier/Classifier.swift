//
//  Classifier.swift
//  
//
//  Created by Sylvan Martin on 6/28/22.
//

import Foundation
import MatrixKit

/**
 * A neural network used to classify inputs into a finite number of possible objects, all of type `ClassificationType`
 */
public class Classifier<T> {
    
    // MARK: Properties
    
    /**
     * The underlying neural network for this classifier
     */
    private let network: NeuralNetwork
    
    /**
     * The way of converting an output layer of neurons into a classification
     *
     * - Precondition: The matrix passed to `classify` will always be a vector of the appropriate length
     */
    public let interpret: (Matrix) -> T
    
    // MARK: Initializers
    
    /**
     * Creates a classifier from a neural network and classification method.
     *
     * - Parameter network: The neural network to use for classification
     * - Parameter classify: A closure used to interpret the output of the neural network and convert it to a `T`
     */
    public init(fromNetwork network: NeuralNetwork, withClassificationMethod classify: @escaping (Matrix) -> T) {
        self.network = network
        self.interpret = classify
    }
    
    // MARK: Methods
    
    /**
     * Classifies an input
     *
     * - Parameter input: Input data to classify
     *
     * - Returns: This classifiers best classification attempt
     */
    public func classify(_ input: [Double]) -> T {
        classify(Matrix(vector: input))
    }
    
    /**
     * Classifies an input vector
     *
     * - Parameter input: An input vector to classify
     *
     * - Returns: This classifiers best classification attempt
     */
    public func classify(_ vector: Matrix) -> T {
        interpret(network.computeOutputLayer(forInput: vector))
    }
    
    
}
