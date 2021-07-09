//
//  NeuralNetwork.swift
//  
//
//  Created by Sylvan Martin on 6/29/21.
//

import Foundation
import MatrixKit
import Accelerate

/**
 * A deep learning model using a neural netowkr
 */
@available(macOS 10.12, *)
class NeuralNetwork {
    
    // MARK: Properties
    
    /**
     * The list of all weights in the network between layers.
     */
    var weights: [Matrix]
    
    /**
     * A list of vectors, each with a bias for a node in the network.
     */
    var biases: [Matrix]
    
    /**
     * The compression function this network uses to calculate its activations
     */
    var activationFunction: ActivationFunction
    
    /**
     * The number of layers in this network
     *
     * This is computed just by adding 1 to the number of bias vectors
     */
    var layers: Int {
        return biases.count + 1
    }
    
    // MARK: Initializers
    
    /**
     * Creates a neural network of a dimension
     *
     * - Parameters:
     *      - layerSizes: Array of the layer sizes. For example, `[3, 4, 4, 2]` will produce a network with an input layer of size 3, two intermediate layers of size 4, and an output layer of size 2.
     *      - activationFunction: Neuron activation function
     *      - random: If `true`, this network will be generated with random weights and biases
     */
    init(layerSizes: [Int], activationFunction: ActivationFunction = .rectifiedLinear, random: Bool = true) {
        self.biases  = [Matrix](repeating: Matrix(), count: layerSizes.count - 1)
        self.weights = [Matrix](repeating: Matrix(), count: layerSizes.count - 1)
        
        let rng: () -> Double = random ? Double.random : { 0 }
        
        for i in 1..<layerSizes.count {
            biases[i - 1]   = Matrix.random(rows: layerSizes[i], cols: 1, rng: rng)
            weights[i - 1]  = Matrix.random(rows: layerSizes[i], cols: layerSizes[i - 1], rng: rng)
        }
        
        self.activationFunction = activationFunction
    }
    
    /**
     * Decodes a `Data` object to a `NeuralNetwork`
     *
     * - Parameters:
     *      - data: `Data` object representing an encoded Neural Network in the proper format.
     */
    convenience init(fromData data: Data) {
        
        let bytesToDecode = data.withUnsafeBytes { $0.bindMemory(to: UInt8.self) }
        
        // The size of an integer on the machine that encoded the data
        let remoteIntSize = Int(bytesToDecode[0])
        
        // The size of a double should be 8 bytes on all machines.
        let doubleSize = MemoryLayout<Double>.size
        
        // This is just a counter to keep track of where we are in the array.
        var byteIndex = remoteIntSize
        
        // The number of layers in the encoded Neural Network
        let numberOfLayers = bytesToDecode[1...byteIndex].withUnsafeBytes {
            $0.bindMemory(to: Int.self)
        }.first!
        
        byteIndex += 1
        
        let lastDimensionIndex = byteIndex + remoteIntSize * numberOfLayers
        
        // get the dimensions of the encoded network
        let dimensions = Array(bytesToDecode[byteIndex..<lastDimensionIndex].withUnsafeBytes {
            $0.bindMemory(to: Int.self)
        })
        
        byteIndex = lastDimensionIndex
        
        // Go ahead and initialize from the dimensions we have to set up the weights and biases
        self.init(layerSizes: dimensions, random: false)
        
        // Now populate the weights and biases
        
        for i in 0..<weights.count {
            let lastWeightIndex = byteIndex + doubleSize * weights[i].flatmap.count
            
            weights[i].flatmap = Array(bytesToDecode[byteIndex..<lastWeightIndex].withUnsafeBytes {
                $0.bindMemory(to: Double.self)
            })
            
            byteIndex = lastWeightIndex
        }
        
        for i in 0..<biases.count {
            let lastBiasIndex = byteIndex + doubleSize * biases[i].flatmap.count
            
            biases[i].flatmap = Array(bytesToDecode[byteIndex..<lastBiasIndex].withUnsafeBytes {
                $0.bindMemory(to: Double.self)
            })
            
            byteIndex = lastBiasIndex
        }
        
        // uh, I think that's it?
        
    }
    
    // MARK: Calculation
    
    /**
     * Computes the output vector of this network given an input vector
     *
     * - Precondition: `input` is a vector, not a matrix
     *
     * - Parameters:
     *      - input: A vector for the activations of the input layer
     */
    func compute(from input: Matrix) -> Matrix {
        
        var activations = input
        
        // we may need to transpose the activations if it is oriented the wrong way.
        if activations.cols > activations.rows {
            activations = activations.transpose
        }
        
        for i in 0..<biases.count {
            activations.applyTransformation(weights[i])
            activations.add(biases[i])
            activations.applyToAllElements(activationFunction.function)
        }
        
        return activations
        
    }
    
}
