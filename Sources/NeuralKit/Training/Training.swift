//
//  File.swift
//  
//
//  Created by Sylvan Martin on 6/30/21.
//

import Foundation
import MatrixKit

/**
 * A class to handle all training of a neural network
 */
@available(macOS 10.12, *)
class NNTrainer {
    
    /**
     * A single item of data to use for training or testing
     */
    struct NNTrainingItem {
        var input: Matrix
        var output: Matrix
        
        init(input: [Double], output: [Double]) {
            self.input  = Matrix(fromCols: [input])!
            self.output = Matrix(fromCols: [output])!
        }
        
        init(input: Matrix, output: Matrix) {
            self.input  = input
            self.output = output
        }
    }
    
    // MARK: Properties
    
    /**
     * The network to train
     */
    var network: NeuralNetwork
    
    /**
     * Pieces of data specifically used for training
     */
    var trainingItems: [NNTrainingItem]
    
    /**
     * Training items specifically used for testing
     */
    var testingItems: [NNTrainingItem]
    
    // MARK: Initializers
    
    init(_ network: NeuralNetwork, trainingItems: [NNTrainingItem] = [], testingItems: [NNTrainingItem] = []) {
        self.network = network
        self.trainingItems = trainingItems
        self.testingItems = testingItems
    }
    
    // MARK: Training Data Organization
    
    /**
     * Takes a percent of training items and allocates them to be used as testing data instead.
     *
     * - Parameters:
     *      - percentTraining: Percent of all training items that should be used as testing items. Default is 10%
     */
    func splitTrainingAndTestingData(percentTraining: Double = 0.1) {
        if trainingItems.isEmpty { return }
        
        // If there is any testing data already existing, add it to the training data just to re-mix it
        trainingItems += testingItems
        
        trainingItems.shuffle()
        
        let numberOfItems = Int(ceil(percentTraining * Double(trainingItems.count)))
        
        testingItems = trainingItems.suffix(numberOfItems)
        trainingItems.removeLast(numberOfItems)
    }
    
    /**
     * Adds or loads training data
     *
     * - Parameters:
     *      - trainingItems: Array of `NNTrainingItem`s to add to this trainer's data
     *      - isTestingSet: Set this to `true` if this data should be loaded as testing data instead of training data
     *      - shouldOverwrite: Set this to `true` if you wish to erase all current training data and load new data
     */
    func load(trainingItems: [NNTrainingItem], isTestingSet: Bool = false, shouldOverwrite: Bool = false) {
        
        if shouldOverwrite {
            if isTestingSet {
                self.testingItems = []
            } else {
                self.trainingItems = []
            }
        }
        
        if isTestingSet {
            self.testingItems += trainingItems
        } else {
            self.trainingItems += trainingItems
        }
        
    }
    
    /**
     * Adds or loads training data
     *
     * This will fail if given an unequal amount of inputs and outputs. This only means the sizes of the arrays must be equal. They may have different vector dimensions.
     *
     * - Parameters:
     *      - trainingInputs: Array of vectors representing known inputs
     *      - isTestingSet: Array of vectors representing known outputs, corresponding to the inputs.
     *      - shouldOverwrite: Set this to `true` if you wish to erase all current training data and load new data
     */
    func load(trainingInputs: [Matrix],  trainingOutputs: [Matrix], isTestingSet: Bool = false, shouldOverwrite: Bool = false) {
        
        let newItems = zip(trainingInputs, trainingOutputs).map { input, output in
            NNTrainingItem(input: input, output: output)
        }
        
        self.load(trainingItems: newItems, isTestingSet: isTestingSet, shouldOverwrite: shouldOverwrite)
    }
    
    /**
     * Adds or loads training data
     *
     * This will fail if given an unequal amount of inputs and outputs. This only means the sizes of the arrays must be equal.
     *
     * - Parameters:
     *      - trainingInputs: Array of sets of input values representing known inputs
     *      - isTestingSet: Array of sets of output values representing known outputs, corresponding to the inputs.
     *      - shouldOverwrite: Set this to `true` if you wish to erase all current training data and load new data
     */
    func load(trainingInputs: [[Double]],  trainingOutputs: [[Double]], isTestingSet: Bool = false, shouldOverwrite: Bool = false) {
        
        let inputMatrices = trainingInputs.map { column in
            Matrix(fromCols: [column])!
        }
        
        let outputMatrices = trainingOutputs.map { column in
            Matrix(fromCols: [column])!
        }
        
        self.load(trainingInputs: inputMatrices, trainingOutputs: outputMatrices, isTestingSet: isTestingSet, shouldOverwrite: shouldOverwrite)
    }
    
    // MARK: Evaluation Functions
    
    /**
     * Computes the cost for a single training sample
     *
     * - Parameters:
     *      - sample: Training sample to evaluate the network based on
     */
    func cost(for sample: NNTrainingItem) -> Double {
        let networkGuess = network.compute(from: sample.input)
        let squareDifferences = (networkGuess - sample.output).applyingToAllElements { pow($0, 2) }
        let sum = squareDifferences.getCol(0).reduce(0, +)
        return sum
    }
    
    /**
     * Computes the average cost for all training data
     */
    func averageCost() -> Double {
        let sum = trainingItems.map { sample in
            cost(for: sample)
        }.reduce(0, +)
        
        let average = sum / Double(trainingItems.count)
        return average
    }
    
}
