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
    
    /**
     * Computes the gradients, in weight space, of the cost function for a particular training example.
     *
     * This is just backpropogation for the weights.
     *
     * - Parameter network: The network to compute the cost of
     * - Parameter example: The training example to compute the gradient for
     * - Parameter gradients: The array of weight gradients to be "filled" by this function
     * - Parameter normalizingGradient: If set to `true`, the gradient vector will be normalized before being applied to each weight.
     *
     * - Precondition: `gradients.count == network.weights.count`
     */
    public static func computeGradients(ofNetwork network: NeuralNetwork, forExample example: DataSet.Item, weightGradients: inout [Matrix], biasGradients: inout [Matrix], normalizingGradient: Bool = true) {
        
        // all the comments here are to keep things straight in my mind, because there's a lot of
        // room to make an off-by-one error here.
        
        // just initialize all these to empty arrays, this only works because Matrix is a value type and MUST STAY THAT WAY PLEASE
        var activations = [Matrix](repeating: Matrix(), count: network.layerCount)
    
        // these are the computed derivatives for each layer EXCEPT the first. Therefore,
        // derivatives[i] refers to the derivative vector for the (i + 1)-th layer
        var derivatives = [Matrix](repeating: Matrix(), count: network.layerCount - 1)
        
        // these partial activations are the input to the activation function for each layer EXCEPT the first one,
        // like the derivatives array.
        var partials = derivatives
        
        network.feedForward(input: example.input, cache: &activations, beforeAdjustedCache: &derivatives)
        
        // The cost gradient for the last layer, in activation space
        var costGradient = activations.last!
        
        // Derivative of cost function
        costGradient -= example.output
        costGradient *= 2
        
        // compute all the derivatives at the pre-activated neuron value (before act. func. applied)
        for i in 0..<derivatives.count {
            derivatives[i].applyToAll(network.activationFunction.applyDerivative)
        }
        
        // compute the partials and gradient for the last layer so the rest are easy
        partials[partials.count - 1] = derivatives[derivatives.count - 1].hadamard(with: costGradient)
        weightGradients[weightGradients.count - 1] = partials[partials.count - 1] * activations[activations.count - 2].transpose
        
        // biases are no different than a weight with a constant input of one
        
        backprop(
            layer: partials.count - 2,
            network: network,
            activations: &activations,
            derivatives: &derivatives,
            weightGradients: &weightGradients,
            biasGradients: &biasGradients,
            partials: &partials
        )
        
        if normalizingGradient {
            for i in 0..<weightGradients.count {
                weightGradients[i].normalize()
            }
        }
    }
    
    private static func backprop(layer: Int, network: NeuralNetwork, activations: inout [Matrix], derivatives: inout [Matrix], weightGradients: inout [Matrix], biasGradients: inout [Matrix], partials: inout [Matrix]) {
        if layer == 0 { return }
        
        partials[i] = derivatives[i].hadamard(with: network.weights[i + 1].transpose * partials[i + 1])
        weightGradients[i] = activations[i].transpose.leftMultiply(by: partials[i])
        
        backprop(
            layer: layer - 1,
             network: network,
             activations: &activations,
             derivatives: &derivatives,
             weightGradients: &weightGradients,
             biasGradients: &biasGradients,
             partials: &partials
        )
    }
    
    /**
     * Applies one step of gradient descent to the weights of a neural network
     */
    public static func performStep(on network: NeuralNetwork, forExample example: DataSet.Item, learningRate: Double, normalizingGradient: Bool = true) {
        var weightGradients = network.weights.map { $0.zero }
        var biasGradients = network.biases.map { $0.zero }
    
        GradientDescent.computeGradients(ofNetwork: network, forExample: example, weightGradients: &weightGradients, biasGradients: &biasGradients, normalizingGradient: normalizingGradient)
        
        for i in 0..<weightGradients.count {
            
            // I still have no idea why this makes sense, because in theory we should be subtracting the gradient,
            // so clearly I got a sign flipped somewhere but I cannot find it, and this seems to work.
            
//            network.weights[i].subtract(learningRate * weightGradients[i])
            network.weights[i].add(learningRate * weightGradients[i])
            
//            network.biases[i].subtract(learningRate * biasGradients[i])
            network.biases[i].add(learningRate * biasGradients[i])
        
        }
    }
    
}
