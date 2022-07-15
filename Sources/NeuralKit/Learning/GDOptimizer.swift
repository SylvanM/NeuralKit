//
//  GDOptimizer.swift
//  
//
//  Created by Sylvan Martin on 7/12/22.
//

import Foundation
import MatrixKit

public class GDOptimizer {
    
    // MARK: Optimization Methods
    
    
    
    /**
     * Optimizes a network using all training examples
     *
     * - Parameter network: The `NeuralNetwork` to optimize
     * - Parameter learningRate: The size of each step taken down the gradient
     * - Parameter dataSet: The `DataSet` to use as training data
     * - Parameter normalizingGradient: If set to `true`, the gradient of the cost function will be normalized before being applied to the weights and biases.
     */
    public static func optimize(_ network: NeuralNetwork, learningRate: Double, forDataSet dataSet: DataSet, normalizingGradient: Bool = true) {
        dataSet.iterateTrainingData { example in
            GradientDescent.performStep(on: network, forExample: example, learningRate: learningRate, normalizingGradient: normalizingGradient)
        }
    }
    
}
