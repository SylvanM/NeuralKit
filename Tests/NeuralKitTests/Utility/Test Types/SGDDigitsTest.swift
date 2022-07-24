//
//  SGDDigitsTest.swift
//  
//
//  Created by Sylvan Martin on 7/23/22.
//

import XCTest
import NeuralKit

class SGDDigitsTest: SGDTest {
    
    override func run() {
        super.run()
        
        let digitClassifier = Classifier(fromNetwork: network) { output -> Int in
            var digit = 0
            for i in 1...9 {
                if output[i] > output[digit] {
                    digit = i
                }
            }
            return digit
        }
        
        var correct = 0
        var counter = 0
        
        DataSet.mnist.iterateTestingData { item in
            
            let label = digitClassifier.interpret(item.output)
            let computed = digitClassifier.classify(item.input)
            
            if label == computed {
                correct += 1
            }
            
            counter += 1
        }
        
        let accuracy = 100 * Double(correct) / Double(DataSet.mnist.testingItemsCount)
        
        print("Final accuracy: \(String(format: "%.02f", accuracy))")
    }

}
