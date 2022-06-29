//
//  NeuralNetwork.swift
//  
//
//  Created by Sylvan Martin on 6/28/22.
//

import Foundation
import Accelerate
import MatrixKit

public class NeuralNetwork {
    
    // MARK: Properties
    
    /**
     * - Invariant: Something about how the weights and biases have to actually work together to
     * form a legit neural network
     */
    
    /**
     * The weight matrices for each layer of the neural network
     *
     * - Invariant: All these matrices can be multiplied together! (that's informal sorry)
     */
    var weights: [Matrix] // TODO: Formalize this documentation
    
    /**
     * Bias vectors for each layer of the neural network, minus the first layer.
     */
    var biases: [Matrix]
    
    /**
     * The activation function for the neural network
     */
    var activationFunction: ActivationFunction
    
    // MARK: Initializers
    
    /**
     * Creates a network from certain weights and biases
     *
     * - Precondition: These satisfy the invariant
     *
     * - Parameter weights: The weights for the neural network
     * - Parameter biases: The biases for each activation layer other than the input layer.
     */
    init(weights: [Matrix], biases: [Matrix], activationFunction: ActivationFunction) {
        self.weights = weights
        self.biases = biases
        self.activationFunction = activationFunction
    }
    
    // MARK: Methods
    
    /**
     * Computes the output layer for a given input layer
     */
    public func computeOutputLayer(forInput input: Matrix) -> Matrix {
        compute(layer: weights.count, forInput: input)
    }
    
    internal func compute(layer: Int, forInput input: Matrix) -> Matrix {
        var currentLayer = input
        
        for i in 0..<layer {
            currentLayer = weights[i] * currentLayer
            currentLayer.add(biases[i])
            currentLayer.applyToAll(activationFunction.apply)
        }
        
        return currentLayer
    }
    
}
