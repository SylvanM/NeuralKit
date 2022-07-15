//
//  GDTestsXOR.swift
//
//
//  Created by Sylvan Martin on 7/12/22.
//

import XCTest
import NeuralKit
import MatrixKit

class GDTestsXOR: XCTestCase {
    
    func makeNetwork() -> NeuralNetwork {
        NeuralNetwork(randomWithShape: [2, 2, 1], withBiases: false, activationFunction: .sigmoid)
    }
    
    let xorDataSet = try! DataSet(name: "XOR", inDirectory: URL(fileURLWithPath: "/Users/sylvanm/Programming/Machine Learning/Data sets/NKDS Sets/XOR Data Set"))
    

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }
    
    func computingGradient() throws {
        
        let XORNetwork = makeNetwork()
        
        xorDataSet.iterateTestingData { item in
            print("--------------------")
            print("Example:")
            print(XORItem(item))
            print()
            
            var weightGrads = [Matrix](repeating: Matrix(), count: XORNetwork.weights.count)
            
            GradientDescent.computeWeightGradients(ofNetwork: XORNetwork, forExample: item, gradients: &weightGrads)
            
            for i in 0..<weightGrads.count {
                print("Weight gradient for weight layer \(i + 1):")
                print(weightGrads[i])
            }
        }
        
    }
    
    func testEachSGDStep() throws {
        
        // made to ensure that each step does in fact reduce the cost
        let XORNetwork = makeNetwork()
        
        let learningRate = 0.05
        
        xorDataSet.iterateTrainingData { trainingItem in
            let beforeStepCost = XORNetwork.cost(for: trainingItem)
            
            let beforeInv = XORNetwork.invariantSatisied
            
            GradientDescent.performStep(on: XORNetwork, forExample: trainingItem, learningRate: learningRate)
            
            let afterInv = XORNetwork.invariantSatisied
            
            if !afterInv || !beforeInv {
                XCTFail()
            }
            
            let afterStepCost = XORNetwork.cost(for: trainingItem)
            
            XCTAssertLessThan(afterStepCost, beforeStepCost)
        }
        
        
    }
    
    func testFullSGDOptimization() throws {
    
        let XORNetwork = makeNetwork()
        
        let learningRate: Double = 1
        
        print("Beginning optimization")
        GDOptimizer.optimize(XORNetwork, learningRate: learningRate, forDataSet: xorDataSet)
        
        let testingCost = xorDataSet.testingCost(of: XORNetwork)
        
        let xorClassifier = Classifier(fromNetwork: XORNetwork) { output -> Double in
            output[0]
        }
        
        print("Final testing cost: \(testingCost)")
        print("Showing some examples...")
        
        xorDataSet.iterateTestingData { item in
            
            let xorItem = XORItem(item)
            let choice = xorClassifier.classify(xorItem.input)
            
            print("--------------------------------")
            print(xorItem)
            print("Network says: \(choice)")
            
        }
        
    }

}
