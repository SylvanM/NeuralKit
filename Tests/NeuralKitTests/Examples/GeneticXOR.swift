//
//  GeneticAlgorithmExamples.swift
//  
//
//  Created by Sylvan Martin on 7/1/22.
//

import XCTest
import NeuralKit
import MatrixKit

/**
 * A test suite demonstrating training a neural network to act as an XOR logic gate using a genetic algorithm
 */
class GeneticXOR: XCTestCase {
    
    // MARK: Examples
    
    func testWriteXORDataSet() throws {
        let inputSpace = [0, 1]
        let trainingData = inputSpace.map { lhs in
            inputSpace.map { rhs in
                DataSet.Item(input: Matrix(vector: [Double(lhs), Double(rhs)]), output: Matrix(vector: [Double(lhs ^ rhs)]))
            }
        }.reduce([], +)
        
        let directory = URL(fileURLWithPath: "/Users/sylvanm/Programming/Machine Learning/Data sets/NKDS Sets/XOR Data Set")
        _ = try DataSet(name: "XOR", trainingItems: trainingData, testingItems: trainingData, inDirectory: directory)
        
    }
    
    func testGeneticOptimizer() {
        measure {
            let xorShape = [2, 2, 1]
            let activationFunction = ActivationFunction.step
            let generationSize = 1000
            let iterations = 500
            let fitnessCutoff = 0.7
            
            let directory = URL(fileURLWithPath: "/Users/sylvanm/Programming/Machine Learning/Data sets/NKDS Sets/XOR Data Set")
            let dataSet = try! DataSet(name: "XOR", inDirectory: directory)
            
            let optimizer = GeneticOptimizer(optimizingShape: xorShape, activationFunction: activationFunction, withData: dataSet, breedingMethod: .arithmeticMean())
            
            let finalNetwork = optimizer.findOptimalNetwork(populationSize: generationSize, eliminatingPortion: fitnessCutoff, iterations: iterations)
            let finalNetworkCost = dataSet.trainingCost(of: finalNetwork)
            
            print("Final network cost: \(finalNetworkCost)")
            
            XCTAssertEqual(finalNetworkCost, 0)
            
            print("Showing classifications of the optimized network:")
            let inputSpace = [0, 1]
            inputSpace.forEach { a in
                inputSpace.forEach { b in
                    let computed = finalNetwork.computeOutputLayer(forInput: Matrix(vector: [Double(a), Double(b)]))
                    print("\(a) ^ \(b) = \(computed)")
                    XCTAssertEqual(a ^ b, Int(computed[0, 0]))
                }
            }
        }
    }
    
    func testGenericAlg() throws {
        
        let directory = URL(fileURLWithPath: "/Users/sylvanm/Programming/Machine Learning/Data sets/NKDS Sets/XOR Data Set")
        let dataSet = try DataSet(name: "XOR", inDirectory: directory)
        
        let inputSpace = [0, 1]
        
        func score(_ net: NeuralNetwork) -> Double {
            -dataSet.trainingCost(of: net)
        }
        
        func avgScore(_ population: [NeuralNetwork]) -> Double {
            population.reduce(into: 0) { partialResult, net in
                partialResult += score(net)
            } / Double(population.count)
        }
        
        // Now we actually make the population!
        
        let xorShape = [2, 2, 1]
        let activationFunction = ActivationFunction.step
        let generationSize = 1000
        let iterations = 50
        let fitnessCutoff = 0.7
        
        var population = [NeuralNetwork](repeating: NeuralNetwork(shape: xorShape), count: generationSize)
        
        var scores = [GenericGeneticAlgorithm<NeuralNetwork>.ScoreRecord](repeating: (index: 0, score: 0), count: population.count)
        
        for i in 0..<population.count {
            population[i] = NeuralNetwork(randomWithShape: xorShape, activationFunction: activationFunction, weightRange: -10...10, biasRange: -10...10)
        }
        
        let initialScore = avgScore(population)
        
        print("Initial score: \(initialScore)")
        
        var counter = 0
        
        GenericGeneticAlgorithm.evolve(
            organisms: &population,
            scores: &scores,
            fitness: fitnessCutoff,
            generations: iterations,
            score: score,
            breed: GenericGeneticAlgorithm.BreedingFunction.arithmeticMean().breed
        ) {
            counter += 1
            if counter % 100 == 0 {
                print("Finished generation \(counter)")
            }
        }
        
        let finalScore = avgScore(population)
        
        XCTAssertGreaterThan(finalScore, initialScore)
        
        print("Final Score: \(finalScore)")
        print("Improvement: \(finalScore - initialScore)")
        
        scores.sort { lhs, rhs in
            lhs.score > rhs.score
        }
        
        let sampleSize = 10
        let sampleIndices = scores[0..<sampleSize].map { $0.index }
        
        print("Showcasing the top sample of \(sampleSize) talented survivors:")
        
        print("------------------------------")
        for i in 0..<sampleIndices.count {
            let net = population[sampleIndices[i]]
            
            print("According to survivor \(sampleIndices[i]) with score \(score(net)),")
            
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
