//
//  GradientDescent.swift
//  
//
//  Created by Sylvan Martin on 7/12/22.
//

import Foundation
import MatrixKit

/**
 * The algorithm for computing the optimization step in training a neural network
 */
public class GradientDescent {
    
    // MARK: Properties
    
    /**
     * The network to perform gradient decent on
     */
    public var network: NeuralNetwork
    
    /**
     * Whether or not the gradient should be normalized
     */
    public var shouldNormalizeGradient: Bool
    
    /**
     * The data set to use to train this network
     */
    public var dataSet: DataSet
    
    /**
     * The learning rate
     */
    public var learningRate: Double
    
    /**
     * `true` if the biases should be optimized
     */
    public var shouldOptimizeBiases: Bool
    
    // MARK: Initializers
    
    /**
     * Creates a `GradientDescent` optimizer
     *
     * - Parameter network: The `NeuralNetwork` to analyze
     * - Parameter dataSet: The `DataSet` to use for optimization
     * - Parameter learningRate: The learning rate (sorry)
     * - Parameter shouldNormalizeGradient: If `true`, the computed gradient will be normalized
     */
    public init(for network: NeuralNetwork, usingDataSet dataSet: DataSet, learningRate: Double, shouldNormalizeGradient: Bool = true, shouldOptimizeBiases: Bool = true) {
        self.network = network
        self.dataSet = dataSet
        self.learningRate = learningRate
        self.shouldNormalizeGradient = shouldNormalizeGradient
        self.shouldOptimizeBiases = shouldOptimizeBiases
    }
    
    /**
     * Computes the gradients, in weight space, of the cost function for a particular training example.
     *
     * This is just backpropogation for the weights.
     *
     * - Parameter example: The training example to compute the gradient for
     * - Parameter weightGradients: The array of weight gradients to be "filled" by this function
     * - Parameter biasGradients: The array of bias gradients to be computed by this function
     *
     * - Precondition: `weightGradients.count == network.weights.count`
     * - Precondition: `biasGradients.count == network.baises.count`
     */
    public func computeGradients(forExample example: DataSet.Item, weightGradients: inout [Matrix], biasGradients: inout [Matrix]) {
        
        // all the comments here are to keep things straight in my mind, because there's a lot of
        // room to make an off-by-one error here.
        
        // just initialize all these to empty arrays, this only works because Matrix is a value type and MUST STAY THAT WAY PLEASE
        var activations = [Matrix](repeating: Matrix(), count: network.layerCount)
    
        // these are the computed derivatives for each layer EXCEPT the first. Therefore,
        // derivatives[i] refers to the derivative vector for the (i + 1)-th layer
        var derivatives = [Matrix](repeating: Matrix(), count: network.layerCount - 1)
        
        network.feedForward(input: example.input, cache: &activations, beforeAdjustedCache: &derivatives)
        
        // The cost gradient for the last layer, in activation space
        var costGradient = activations.last!
        
        // Derivative of cost function
        costGradient -= example.output
        costGradient *= 2
        
        // compute all the derivatives at the pre-activated neuron value (before act. func. applied)
        for i in 0..<derivatives.count {
            derivatives[i].applyToAll(network.activationFunctions[i].applyDerivative)
        }
        
        var originalPartial = derivatives[derivatives.count - 1].hadamard(with: costGradient)
        
        backprop(
            layer: derivatives.count - 1,
            network: network,
            activations: &activations,
            derivatives: &derivatives,
            weightGradients: &weightGradients,
            biasGradients: &biasGradients,
            partial: &originalPartial
        )
        
        if shouldNormalizeGradient {
            for i in 0..<weightGradients.count {
                weightGradients[i].normalize()
                biasGradients[i].normalize()
            }
        }
    }
    
    /**
     * - Parameter layer: The layer to write to
     * - Parameter network: The network to use
     * - Parameter activations: The activations of layer `layer`
     * - Parameter derivatives: The derivatives of the activations for layer `layer`
     * - Parameter weightGradients: The weight gradients to write
     * - Parameter biasGradients: The bias gradients to write
     * - Parameter partial: The partial derivative to use in this layer's calculation
     */
    private func backprop(layer: Int, network: NeuralNetwork, activations: inout [Matrix], derivatives: inout [Matrix], weightGradients: inout [Matrix], biasGradients: inout [Matrix], partial: inout Matrix) {
        if layer < 0 { return }
        
        weightGradients[layer] = activations[layer].transpose.leftMultiply(by: partial)
        
        biasGradients[layer] = partial
        
        if layer > 0 {
            partial = derivatives[layer - 1].hadamard(with: network.weights[layer].transpose * partial)
        }
        
        backprop(
            layer: layer - 1,
            network: network,
            activations: &activations,
            derivatives: &derivatives,
            weightGradients: &weightGradients,
            biasGradients: &biasGradients,
            partial: &partial
        )
    }
    
    /**
     * Applies one step of gradient descent to the weights of a neural network
     */
    public func performStep(forExample example: DataSet.Item) {
        var weightGradients = network.weights.map { $0.zero }
        var biasGradients   = network.biases.map  { $0.zero }
    
        computeGradients(forExample: example, weightGradients: &weightGradients, biasGradients: &biasGradients)
        
        for i in 0..<weightGradients.count {
            
            // I still have no idea why this makes sense, because in theory we should be subtracting the gradient,
            // so clearly I got a sign flipped somewhere but I cannot find it, and this seems to work.
            
//            network.weights[i].subtract(learningRate * weightGradients[i])
            network.weights[i].add(learningRate * weightGradients[i])
            
            if shouldOptimizeBiases {
//                network.biases[i].subtract(learningRate * biasGradients[i])
                network.biases[i].add(learningRate * biasGradients[i])
            }

        }
    }
    
    /**
     * Applies a step of backpropogation for every training item in a data set
     */
    public func optimize() {
        dataSet.iterateTrainingData {
            performStep(forExample: $0)
        }
    }
    
    
}
