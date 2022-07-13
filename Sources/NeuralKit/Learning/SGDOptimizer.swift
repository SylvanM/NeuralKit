//
//  SGDOptimizer.swift
//  
//
//  Created by Sylvan Martin on 7/12/22.
//

import Foundation
import MatrixKit

public class SGDOptimizer {
    
    // MARK: Optimization Methods
    
    /**
     * Applies one step of gradient descent to the weights of a neural network
     */
    public static func performStep(on network: NeuralNetwork, forExample example: DataSet.Item, learningRate: Double) {
        var gradients = [Matrix](repeating: Matrix(), count: network.weights.count)
        GradientDescent.computeWeightGradients(ofNetwork: network, forExample: example, gradients: &gradients)
        
        for i in 0..<gradients.count {
            network.weights[i].subtract(learningRate * gradients[i])
        }
    }
    
    /**
     * Optimizes a network using all training examples
     *
     * - Parameter network: The `NeuralNetwork` to optimize
     * - Parameter learningRate: The size of each step taken down the gradient
     * - Parameter dataSet: The `DataSet` to use as training data
     */
    public static func optimize(_ network: NeuralNetwork, learningRate: Double, forDataSet dataSet: DataSet) {
        dataSet.iterateTrainingData { example in
            performStep(on: network, forExample: example, learningRate: learningRate)
        }
    }
    
}
