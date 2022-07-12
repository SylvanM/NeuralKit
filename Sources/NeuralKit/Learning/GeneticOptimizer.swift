//
//  GeneticOptimizer.swift
//  
//
//  Created by Sylvan Martin on 7/5/22.
//

import Foundation
import MatrixKit
import AppKit

/**
 * An optimizer that uses a genetic algorithm to optimize a neural network
 */
public class GeneticOptimizer {
    
    public typealias Algorithm = GenericGeneticAlgorithm<NeuralNetwork>
    
    // MARK: Properties
    
    /**
     * The set used to compute the score for each organism
     */
    public let dataSet: DataSet
    
    /**
     * The shape of the neural network to evolve
     */
    public let shape: NeuralNetwork.Shape
    
    /**
     * The activation function of the neural networks to evolve
     */
    public let activationFunction: ActivationFunction
    
    /**
     * The function determining how two networks are bred
     */
    public var breedFunction: Algorithm.BreedingFunction
    
    
    // MARK: Initializers
    
    /**
     * Creates a class that optimizes a neural network using a genetic algorithm
     *
     * - Parameter shape: The shape of the network to find the optimal parameters for
     * - Parameter dataSet: The set of data to use to optimize
     * - Parameter breedingMethod: The function to use to breed two `NeuralNetwork`s to create a new one
     */
    public init(optimizingShape shape: NeuralNetwork.Shape, activationFunction: ActivationFunction, withData dataSet: DataSet, breedingMethod: Algorithm.BreedingFunction) {
        self.shape = shape
        self.dataSet = dataSet
        self.breedFunction = breedingMethod
        self.activationFunction = activationFunction
    }
    
    /**
     * Computes the score of a network
     */
    func score(for network: NeuralNetwork) -> Double {
        1 / dataSet.trainingCost(of: network)
    }
    
    /**
     * Creates a neural network that has been optimized, through genetic evolution, to correctly match the data set.
     */
    public func findOptimalNetwork(populationSize: Int, eliminatingPortion: Double, iterations: Int, breed: Algorithm.BreedingFunction = .arithmeticMean(), initialWeightRange: ClosedRange<Double> = -10...10, initialBiasRange: ClosedRange<Double> = -10...10, uponGenerationCompletion: (([NeuralNetwork]) -> ())? = nil) -> NeuralNetwork {
        
        var population = [NeuralNetwork](repeating: NeuralNetwork(shape: shape), count: populationSize)
        var scores = [Algorithm.ScoreRecord](repeating: (index: 0, score: 0), count: populationSize)
        
        for i in 0..<population.count {
            for j in 0..<population[i].weights.count {
                population[i].weights[j].applyToAll { element in
                    element = Double.random(in: initialWeightRange)
                }
                population[i].biases[j].applyToAll { element in
                    element = Double.random(in: initialBiasRange)
                }
            }
            population[i].activationFunction = activationFunction
        }
        
        Algorithm.evolve(organisms: &population, scores: &scores, fitness: eliminatingPortion, generations: iterations, score: score, breed: breed.breed) {
            uponGenerationCompletion?(population)
        }

        return population[scores.last!.index]
    }
    
}

public extension GenericGeneticAlgorithm where S == NeuralNetwork {
    
    /**
     * A breeding function that takes in two neural networks and produces a child
     */
    @dynamicCallable
    struct BreedingFunction {
        
        /**
         * The actual breeding function
         */
        public let breed: (NeuralNetwork, NeuralNetwork) -> NeuralNetwork
        
        /**
         * The syntactic sugar implementation for directly calling this breeding function
         *
         * - Precondition: `args.count == 2`
         */
        func dynamicallyCall(withArguments args: [NeuralNetwork]) -> NeuralNetwork {
            breed(args[0], args[1])
        }
        
        init(_ breedingFunction: @escaping (NeuralNetwork, NeuralNetwork) -> NeuralNetwork) {
            self.breed = breedingFunction
        }
        
        // MARK: Default Breeding Functions
        
        /**
         * Returns the arithmetic mean of two neural networks, averaging their weights and biases
         */
        public static let arithmeticMeanWithoutMutation = BreedingFunction { mother, father in
            let child = NeuralNetwork(mother)
            
            for i in 0..<child.weights.count {
                child.weights[i] = 0.5 * (mother.weights[i] + father.weights[i])
                child.biases[i] = 0.5 * (mother.biases[i] + father.biases[i])
                
            }
            
            return child
        }
        
        /**
         * Returns the arithmetic mean with some random mutation to some weights and biases.
         *
         * - Parameter mutationFrequency: The fraction of all weights and biases to be mutated
         * - Parameter mutationRange: When a particular
         */
        public static func arithmeticMean(mutationFrequency: Double = 0.2, mutationRange: ClosedRange<Double> = -10...10 ) -> BreedingFunction {
            BreedingFunction { mother, father in
                
                let child = arithmeticMeanWithoutMutation(mother, father)
                
                for i in 0..<child.weights.count {
                    child.weights[i].applyToAll { element in
                        if Double.random(in: 0...1) < mutationFrequency {
                            element *= Double.random(in: mutationRange)
                        }
                    }
                    
                    child.biases[i].applyToAll { element in
                        if Double.random(in: 0...1) < mutationFrequency {
                            element *= Double.random(in: mutationRange)
                        }
                    }
                }
                
                return child
                
            }
        }
        
    }
    
}
