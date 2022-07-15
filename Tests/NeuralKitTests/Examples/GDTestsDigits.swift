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
        NeuralNetwork(randomWithShape: [784, 16, 16, 10], withBiases: false, activationFunction: .sigmoid)
    }
    
    let mnistDataSet = try! DataSet(name: "Digits", inDirectory: URL(fileURLWithPath: "/Users/sylvanm/Programming/Machine Learning/Data sets/NKDS Sets/MNIST Digits"))
    

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }
    
    func computingGradient() throws {
        
        let digitsNetwork = makeNetwork()
        
        var counter = 0
        mnistDataSet.iterateTestingData { item in
            if counter % 5000 == 0 {
                
                print("--------------------")
                print("Example:")
                print(MNISTUtility.MNISTItem(item))
                print()
                
                var weightGrads = [Matrix](repeating: Matrix(), count: digitsNetwork.weights.count)
                
                GradientDescent.computeWeightGradients(ofNetwork: digitsNetwork, forExample: item, gradients: &weightGrads)
                
                for i in 0..<weightGrads.count {
                    print("Weight gradient for weight layer \(i + 1):")
                    print(weightGrads[i])
                }
                
            }
            counter += 1
        }
        
    }
    
    func testEachSGDStep() throws {
        
        // made to ensure that each step does in fact reduce the cost
        let digitsNetwork = makeNetwork()
        
        let learningRate = 0.05
        
        var counter = 0
        
        mnistDataSet.iterateTrainingData { trainingItem in
            let beforeStepCost = digitsNetwork.cost(for: trainingItem)
            
            GradientDescent.performStep(on: digitsNetwork, forExample: trainingItem, learningRate: learningRate)
            
            let afterStepCost = digitsNetwork.cost(for: trainingItem)
            
            XCTAssertLessThan(afterStepCost, beforeStepCost)
            
            counter += 1
        }
        
        
    }
    
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
            
            GDOptimizer.optimize(entries[i].network, learningRate: entries[i].rate, forDataSet: mnistDataSet)
            
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
