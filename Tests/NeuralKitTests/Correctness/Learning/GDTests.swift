//
//  GDTests.swift
//  
//
//  Created by Sylvan Martin on 7/23/22.
//

import XCTest
import NeuralKit
import MatrixKit

class GDTests: XCTestCase {
    
    func testXOR() {
        // We're gonna make a neural network and train it!
        
        let dataSet = DataSet.xor
        
        let network = NeuralNetwork(randomWithShape: [2, 2, 1], withBiases: false, activationFunctions: [.sigmoid, .sigmoid])
        
        let optimizer = GradientDescent(for: network, usingDataSet: dataSet, learningRate: 1)
        
        optimizer.optimize(epochs: 5)
        
        print("Finished optimization!")
        
        print("Cost: \(dataSet.testingCost(of: network))")
    }
    
    func testDigits() {
        
        // We're gonna make a neural network and train it!
        
        let dataSet = DataSet.mnist
        
        let network = NeuralNetwork(randomWithShape: [28 * 28, 16, 16, 10], withBiases: false, activationFunctions: [.sigmoid, .sigmoid, .sigmoid])
        
        let optimizer = GradientDescent(for: network, usingDataSet: dataSet, learningRate: 1)
        
        optimizer.optimize(epochs: 5)
        
        print("Finished optimization!")
        
        func vecToDigit(_ matrix: Matrix) -> Int {
            var digit = 0
            for i in 1..<10 {
                if matrix[i] > matrix[digit] {
                    digit = i
                }
            }
            return digit
        }
        
        // Make a classifier, and see how accurate we are!
        let classifier = Classifier(fromNetwork: network, withClassificationMethod: vecToDigit)
        
        // Now find the accuracy on testing data
        
        let totalAmount = dataSet.testingItemsCount
        var correctGuesses = 0
        
        dataSet.iterateTestingData { item in
            let correctDigit = vecToDigit(item.output)
            let guess = classifier.classify(item.input)
            if correctDigit == guess {
                correctGuesses += 1
            } else {
//                print("Classifier FAILED on the input:")
//                print(MNISTUtility.MNISTItem(item))
//                print("Classifiers guess was \(guess), correct was \(correctDigit)")
            }
        }
        
        let accuracy = Double(correctGuesses) / Double(totalAmount) * 100
        
        print("Final Accuracy: \(accuracy)")
        
    }

}
