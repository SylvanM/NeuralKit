//
//  GeneticDigits.swift
//  
//
//  Created by Sylvan Martin on 7/5/22.
//

import XCTest
import NeuralKit

/**
 * A test suite demonstrating training a neural network to recognize "handwritten" digits using a genetic algorithm
 */
class GeneticDigits: XCTestCase {
    
    let digitsDataSet = try! DataSet(name: "Digits", inDirectory: URL(fileURLWithPath: "/Users/sylvanm/Programming/Machine Learning/Data sets/NKDS Sets/MNIST Digits"))
    let nnShape = [784, 16, 16, 10]

    func testGeneticDigits() throws {
        let optimizer = GeneticOptimizer(optimizingShape: nnShape, activationFunction: .sigmoid, withData: digitsDataSet, breedingMethod: .arithmeticMean())
        
        let populationSize = 1000
        let cutoff = 0.5
        let iterations = 5000
        
        let recognizer = optimizer.findOptimalNetwork(populationSize: populationSize, eliminatingPortion: cutoff, iterations: iterations)
        
        let classifier = Classifier(fromNetwork: recognizer) { vector -> Int in
            var digit = 0
            
            for i in 0..<vector.flatmap.count {
                if vector.flatmap[i] > vector.flatmap[digit] {
                    digit = i
                }
            }
            
            return digit
        }
        
        var counter = 0
        digitsDataSet.iterateTestingData { testingItem in
            
            if counter % 1000 == 0 {
                
                print("-------------------------")
                
                let mnistItem = MNISTUtility.MNISTItem(input: testingItem.input, output: testingItem.output)
                
                let decision = classifier.classify(testingItem.input)
                
                print(mnistItem)
                print("Classifier says: \(decision)")
                
            }
            
            counter += 1
        }
    }

}
