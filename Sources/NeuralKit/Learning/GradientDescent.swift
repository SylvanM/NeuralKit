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
        
        // Derivative of cost function
        costGradient -= example.output
        costGradient *= 2
        
        #if DEBUG
        assert(costGradient.flatmap.allSatisfy({ !$0.isNaN }), "Cost Gradient contains NaN")
        #endif
        
        // compute all the derivatives at the pre-activated neuron value (before act. func. applied)
        for i in 0..<derivatives.count {
            derivatives[i].applyToAll(network.activationFunction.applyDerivative)
        }
        
        #if DEBUG
        
        let toAssertNormal = derivatives.allSatisfy { der in
            der.flatmap.allSatisfy {
                !$0.isNaN
            }
        }
        
        let toAssertFininte = derivatives.allSatisfy { der in
            der.flatmap.allSatisfy {
                $0.isFinite
            }
        }
        
        assert(toAssertNormal, "Derivatives contain NaN")
        assert(toAssertFininte, "Derivatives are not all finite")
        
        #endif
        
        // compute the partials and gradient for the last layer so the rest are easy
        partials[partials.count - 1]    = derivatives[derivatives.count - 1].hadamard(with: costGradient)
        
        #if DEBUG
        for i in 0..<partials.count {
            assert(partials[i].flatmap.allSatisfy({ !$0.isNaN }), "Partials contain NaN")
        }
        #endif
        
        gradients[gradients.count - 1]  = partials[partials.count - 1] * activations[activations.count - 2].transpose
        
        for i in (0..<(partials.count - 1)).reversed() {
            partials[i] = derivatives[i].hadamard(with: network.weights[i + 1].transpose * partials[i + 1]) // might need to index weights 1 higher
            
            #if DEBUG
            assert(partials[i].flatmap.allSatisfy({ !$0.isNaN }), "Gradient contains NaN")
            #endif
            
            gradients[i] = activations[i].transpose.leftMultiply(by: partials[i])
        }
        
        
    }
    
    /**
     * Applies one step of gradient descent to the weights of a neural network
     */
    public static func performStep(on network: NeuralNetwork, forExample example: DataSet.Item, learningRate: Double) {
        var gradients = [Matrix](repeating: Matrix(), count: network.weights.count)
    
        GradientDescent.computeWeightGradients(ofNetwork: network, forExample: example, gradients: &gradients)
        
        for i in 0..<gradients.count {
            #if DEBUG

            var foundNan: Bool {
                network.weights[i].flatmap.contains { $0.isNaN }
            }

            if foundNan {
                print("Error: Found NaN in weights BEFORE having computed gradient")
                fatalError()
            }

            #endif
            
            network.weights[i].subtract(learningRate * gradients[i])
            
            #if DEBUG
            
            if foundNan {
                print("Error: Found NaN in weights after having computed gradient")
                fatalError()
            }
            
            #endif
        }
    }
    
}
