//
//  BiasLearningTest.swift
//  
//
//  Created by Sylvan Martin on 7/23/22.
//

import XCTest
import NeuralKit
import MatrixKit

class SGDTest: XCTestCase {
    
    var network: NeuralNetwork!
    var dataSet: DataSet!
    
    var learningRate: Double = 0.5
    var shouldOptimizeBiases = true
    
    override var name: String {
        "SGD Test on \(dataSet.name) with alpha=\(learningRate), optimizing biases: \(shouldOptimizeBiases)"
    }
    
    override func run() {
        print("----------------------------------------------------------------")
        
        let preOptCost = dataSet.trainingCost(of: network)
        
        print(name)
        print("Initial training cost: \(preOptCost)")
        
        let optimizer = GradientDescent(for: network, usingDataSet: dataSet, learningRate: learningRate, shouldOptimizeBiases: shouldOptimizeBiases)
        
        for _ in 1...5 {
            optimizer.optimize()
        }
        
        print("Final cost: \(dataSet.trainingCost(of: network))")
    }
    
    
    
}
