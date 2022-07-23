//
//  GDTestsDigits.swift
//  
//
//  Created by Sylvan Martin on 7/12/22.
//

import XCTest
import NeuralKit
import MatrixKit

class GDTestsDigits: XCTestCase {
    
    func makeNetwork() -> NeuralNetwork {
        let shape = [784, 16, 16, 10]
        let actFuncs: [ActivationFunction] = [.sigmoid, .sigmoid, .sigmoid]
        return NeuralNetwork(randomWithShape: shape, activationFunctions: actFuncs)
    }
    
    let mnistDataSet = try! DataSet(name: "Digits", inDirectory: URL(fileURLWithPath: "/Users/sylvanm/Programming/Machine Learning/Data sets/NKDS Sets/MNIST Digits"))
    
    func testFullGDOptimization() throws {
        
        let learningRates = [0.001, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.2, 1.5, 1.7, 2]
        var entries = learningRates.map { rate in
            (rate: rate, network: makeNetwork(), cost: Double(0))
        }
        
        print("----------------------------------------------------------------")
        
        for i in 0..<entries.count {
            let preOptCost = mnistDataSet.testingCost(of: entries[i].network)
            
            print("Optimizing network \(i) with learning rate \(entries[i].rate)...")
            print("Initial testing cost: \(preOptCost)")
            
            let optimizer = GradientDescent(for: entries[i].network, usingDataSet: mnistDataSet, learningRate: entries[i].rate)
            optimizer.optimize()
            
            entries[i].cost = mnistDataSet.testingCost(of: entries[i].network)
            
            print("Final cost: \(entries[i].cost)")
            print("----------------------------------------------------------------")
        }
        
        entries.sort { lhs, rhs in
            lhs.cost < rhs.cost
        }
        
        print("Optimal learning rate is \(entries[0].rate) with cost \(entries[0].cost), demonstrating optimal network:")
        
        let digitClassifier = Classifier(fromNetwork: entries[0].network) { output -> Int in
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
        
        print("----------------------------------------------------------------")
        
        mnistDataSet.iterateTestingData { item in
            
            let label = digitClassifier.interpret(item.output)
            let computed = digitClassifier.classify(item.input)
            
            if label == computed {
                correct += 1
            }
            
            if counter % (1000 + Int.random(in: -5...5)) == 0 {
                print("Training Example \(counter):")
                print(MNISTUtility.MNISTItem(item))
                print("Network says: \(computed)")
                print("----------------------------------------------------------------")
            }
            
            counter += 1
        }
        
        let accuracy = 100 * Double(correct) / Double(mnistDataSet.testingItemsCount)
        
        print("Final accuracy: \(String(format: "%.02f", accuracy))")
        
    }

}
