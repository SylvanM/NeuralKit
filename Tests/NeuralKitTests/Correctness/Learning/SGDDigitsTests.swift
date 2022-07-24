//
//  SGDDigitsTests.swift
//  
//
//  Created by Sylvan Martin on 7/23/22.
//

import XCTest
import NeuralKit

class SGDDigitsTests: XCTestCase {
    
    // MARK: Helpers
    
    static var baseNetwork: NeuralNetwork = makeNetwork()
    
    static func makeNetwork() -> NeuralNetwork {
        let shape = [784, 16, 16, 10]
        let actFuncs: [ActivationFunction] = [.sigmoid, .sigmoid, .sigmoid]
        return NeuralNetwork(randomWithShape: shape, activationFunctions: actFuncs)
    }
    
    static let learningRates = [0.001, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.2, 1.5, 1.7, 2]
    
    override class var defaultTestSuite: XCTestSuite {
        let suite = XCTestSuite(forTestCaseClass: SGDTest.self)
        
        learningRates.forEach { rate in
            [false, true].forEach { optimizeBiases in
                let sgd = SGDDigitsTest()
                
                sgd.network = NeuralNetwork(baseNetwork)
                sgd.learningRate = rate
                sgd.dataSet = DataSet.mnist
                sgd.shouldOptimizeBiases = optimizeBiases
                
                suite.addTest(sgd)
            }
        }
        
        return suite
    }

}
