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
     * Returns the gradient, in weight space, of the cost function for a particular training example.
     *
     * - Parameter network: The network to compute the cost of
     * - Parameter example: The training example to compute the gradient for
     */
    public static func computeWeightGradient(of network: NeuralNetwork, forExample example: DataSet.Item) -> [Matrix] {
        
        // just initialize all these to empty arrays, this only works because Matrix is a value type and MUST STAY THAT WAY PLEASE
        var activations = [Matrix](repeating: Matrix(), count: network.weights.count + 1)
    
        var derivatives = [Matrix](repeating: Matrix(), count: network.weights.count)
        var partials = activations
        
        var gradients = network.weights
        
        network.feedForward(input: example.input, cache: &activations, beforeAdjustedCache: &derivatives)
        
        // The cost gradient for the last layer, in activation space
        var costGradient = activations.last!
        
        for i in 0..<costGradient.flatmap.count {
            costGradient.flatmap[i] -= example.output.flatmap[i]
            costGradient.flatmap[i] *= 2
        }
        
        for i in 0..<derivatives.count {
            derivatives[i].applyToAll(network.activationFunction.applyDerivative)
        }
        
        // compute the partials and gradient for the last layer so the rest are easy
        partials[partials.count - 1]    = derivatives[partials.count - 1].hadamard(with: costGradient)
        gradients[gradients.count - 1]  = activations[partials.count - 2].leftMultiply(by: partials[partials.count])
        
        for i in (0..<(partials.count - 1)).reversed() {
            partials[i]  = derivatives[i].hadamard(with: network.weights[i]) // might need to index weights 1 higher
            gradients[i] = activations[i].transpose.leftMultiply(by: partials[i])
        }
        
        return gradients
        
    }
    
}
