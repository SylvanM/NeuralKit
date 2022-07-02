//
//  GeneticAlgorithmExamples.swift
//  
//
//  Created by Sylvan Martin on 7/1/22.
//

import XCTest
import NeuralKit
import MatrixKit

class GeneticAlgorithmExamples: XCTestCase {

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    func testExample() throws {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct results.
        // Any test you write for XCTest can be annotated as throws and async.
        // Mark your test throws to produce an unexpected failure when your test encounters an uncaught error.
        // Mark your test async to allow awaiting for asynchronous code to complete. Check the results with assertions afterwards.
    }

    func testPerformanceExample() throws {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }
    
    // MARK: Examples
    
    func testGeneticXOR() {
        
        func score(_ net: NeuralNetwork) -> Double {
            
            var cost: Double = 0
            
            inputSpace.forEach { a in
                inputSpace.forEach { b in
                    let correct = Matrix([[Double(a ^ b)]])
                    let computed = net.computeOutputLayer(forInput: Matrix(vector: [Double(a), Double(b)]))
                    cost += correct.distanceSquared(from: computed)
                }
            }
            
            return -cost
        }
        
        func avgScore(_ population: [NeuralNetwork]) -> Double {
            population.reduce(into: 0) { partialResult, net in
                partialResult += score(net)
            } / Double(population.count)
        }
        
        func breed(_ mother: NeuralNetwork, _ father: NeuralNetwork) -> NeuralNetwork {
            // just take the average of the two
            
            let child = NeuralNetwork(shape: xorShape, activationFunction: activationFunction)
            
            for i in 0..<child.weights.count {
                child.weights[i] = 0.5 * (mother.weights[i] + father.weights[i])
                child.biases[i] = 0.5 * (mother.biases[i] + father.biases[i])
            }
            
            return child
        }
        
        // Now we actually make the population!
        
        let xorShape = [2, 2, 1]
        let activationFunction = ActivationFunction.sigmoid
        let generationSize = 10000
        let iterations = 500
        let fitnessCutoff = 0.7
        
        let inputSpace = [0, 1]
        
        var population = [NeuralNetwork](repeating: NeuralNetwork(shape: xorShape), count: generationSize)
        for i in 0..<population.count {
            population[i] = NeuralNetwork(randomWithShape: xorShape, activationFunction: activationFunction, weightRange: -10...10, biasRange: -10...10)
        }
        
        let initialScore = avgScore(population)
        
        print("Initial score: \(initialScore)")
        
        var counter = 0
        
        measure {
            GeneticTrainer.evolve(
                organisms: &population,
                fitness: fitnessCutoff,
                generations: iterations,
                score: score,
                breed: breed
            ) {
                counter += 1
                if counter % 100 == 0 {
                    print("Finished generation \(counter)")
                }
            }
        }
        
        let finalScore = avgScore(population)
        
        XCTAssertGreaterThan(finalScore, initialScore)
        
        print("Final Score: \(finalScore)")
        print("Improvement: \(finalScore - initialScore)")
        
        let randomSampleSize = 10
        var randomSampleIndices = [Int](repeating: -1, count: randomSampleSize)
        
        for i in 0..<randomSampleSize {
            var randomIndex: Int
            repeat {
                randomIndex = Int.random(in: 0..<population.count)
            } while randomSampleIndices.contains(randomIndex)
            randomSampleIndices[i] = randomIndex
        }
        
        print("Showcasing a random sample of \(randomSampleSize) talented survivors:")
        
        print("------------------------------")
        for i in 0..<randomSampleIndices.count {
            let net = population[randomSampleIndices[i]]
            
            print("According to survivor \(randomSampleIndices[i]) with score \(score(net)),")
            
            inputSpace.forEach { a in
                inputSpace.forEach { b in
                    let computed = net.computeOutputLayer(forInput: Matrix(vector: [Double(a), Double(b)]))
                    print("\(a) ^ \(b) = \(computed)")
                }
            }
            
            print("------------------------------")
        }
        
    }

}
