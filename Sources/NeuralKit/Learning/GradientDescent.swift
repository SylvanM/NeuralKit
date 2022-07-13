//
//  File.swift
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
     *
     * - Precondition: `gradients.count == network.weights.count`
     */
    public static func computeWeightGradients(ofNetwork network: NeuralNetwork, forExample example: DataSet.Item, gradients: inout [Matrix]) {
        
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
        
        for i in 0..<costGradient.flatmap.count {
            costGradient.flatmap[i] -= example.output.flatmap[i]
            costGradient.flatmap[i] *= 2
        }
        
        // compute all the derivatives at the pre-activated neuron value (before act. func. applied)
        for i in 0..<derivatives.count {
            derivatives[i].applyToAll(network.activationFunction.applyDerivative)
        }
        
        // compute the partials and gradient for the last layer so the rest are easy
        partials[partials.count - 1]    = derivatives[derivatives.count - 1].hadamard(with: costGradient)
        gradients[gradients.count - 1]  = activations[partials.count - 2].leftMultiply(by: partials[partials.count - 1])
        
        
        for i in (0..<(partials.count - 1)).reversed() {
            partials[i]  = derivatives[i].hadamard(with: network.weights[i]) // might need to index weights 1 higher
            gradients[i] = activations[i].transpose.leftMultiply(by: partials[i])
        }
        
    }
    
}
