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
    
    func testFullSGDOptimization() throws {
        
        let digitsNetwork = makeNetwork()
        
        let learningRate: Double = 0.7
        
        let finalSampleSize = 10
        
        print("Beginning optimization")
        let preOptCost = mnistDataSet.testingCost(of: digitsNetwork)
        
        print("Initial testing cost: \(preOptCost)")
        
        GDOptimizer.optimize(digitsNetwork, learningRate: learningRate, forDataSet: mnistDataSet)
        let testingCost = mnistDataSet.testingCost(of: digitsNetwork)
        
        print("Final cost: \(testingCost)")
        
        let digitClassifier = Classifier(fromNetwork: digitsNetwork) { output -> Int in
            var digit = 0
            for i in 1...9 {
                if output.flatmap[i] > output.flatmap[digit] {
                    digit = i
                }
            }
            return digit
        }
    
        print("Showing some examples...")
        
        var counter = 0
        
        mnistDataSet.iterateTestingData { item in
            
            if counter <= finalSampleSize {
                let digitItem = MNISTUtility.MNISTItem(item)
                let choice = digitClassifier.classify(digitItem.input)
                
                print("--------------------------------")
                print("Training Example:")
                print(digitItem)
                print("Network says: \(choice)")
            }
            
            counter += 1
            
            // there isn't really a nice way (yet) to only iterate a couple, so we just sit here while
            // it goes through the rest
            
        }
        
    }

}
