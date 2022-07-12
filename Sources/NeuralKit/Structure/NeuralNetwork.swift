//
//  NeuralNetwork.swift
//  
//
//  Created by Sylvan Martin on 6/28/22.
//

import Foundation
import Accelerate
import MatrixKit

/**
 * A graph of layers of nodes containing an activation, where the activation is based off of the activations of the previous layer, with
 * certain nodes having more of an effect than others.
 *
 * This is an unsafe data type. What I mean by that is I am not practicing the usual industry standard of encapsulation. I am delibrerately
 * exposing some parts of implementation and allowing the weights and biases to be directly modified. It is my philosophy
 * that in uses like these, more control to the user is better.
 */
public class NeuralNetwork {
    
    public typealias Shape = [Int]
    
    // MARK: Properties
    
    /**
     * The weight matrices for each layer of the neural network
     *
     * - Invariant: For all `i` in `0..<(weights.count - 1)`, `weights[i + 1].colCount == weights[i].rowCount`. In other words,
     * these matrices can be successively left-multiplied.
     */
    public var weights: [Matrix]
    
    /**
     * Bias vectors for each layer of the neural network, excluding the first layer.
     *
     * If this is null, the network's computation is only based on the weights.
     *
     * - Invariant: `biases.count == weights.count`
     * - Invariant: For all `i` in `0..<biases.count`, `biases[i].colCount == 1`
     * and `biases[i].rowCount == weights[i].rowCount`
     */
    public var biases: [Matrix]
    
    /**
     * The number of layers in this neural network, including the input and output layer
     */
    public var layerCount: Int {
        weights.count + 1
    }
    
    /**
     * This checks that all representation invariants holds.
     *
     * It is an industry standard that properties like this are private, and the client can't even alter the other properties that may violate it at all.
     * However, because I intend for this library to be as versatile as possible, I am delibrately choosing to give the dangerous power of full access
     * to the client. So, I've made this property public as well, so that the client can verify it if they so choose.
     */
    public var invariantSatisied: Bool {
        for i in 0..<(weights.count - 1) {
            if weights[i + 1].colCount != weights[i].rowCount {
                return false
            }
        }
        
        for i in 0..<biases.count {
            if biases[i].colCount != 1 {
                return false
            }
            if biases[i].rowCount != weights[i].rowCount {
                return false
            }
        }
        
        return true
    }
    
    /**
     * The activation function for the neural network.
     *
     * This function is the final step in the computation of the activations for a layer, after the weights and biases have been applied.
     */
    public var activationFunction: ActivationFunction
    
    // MARK: Initializers
    
    /**
     * Creates a network from certain weights and biases
     *
     * - Precondition: These satisfy the invariant
     *
     * - Parameter weights: The weights for the neural network
     * - Parameter biases: The biases for each activation layer other than the input layer.
     * - Parameter activationFunction: The activation function for this neural network.
     */
    public init(weights: [Matrix], biases: [Matrix], activationFunction: ActivationFunction) {
        self.weights = weights
        self.biases = biases
        self.activationFunction = activationFunction
    }
    
    /**
     * Creates an empty neural network of a specific shape
     *
     * - Parameter shape: An array describing the size of each layer. If `shape` is `[2, 2, 1]`, a neural network will be created with
     * 2 input nodes, one hidden layer with 2 nodes, and an output layer with 1 node.
     * - Parameter activationFunction: The activation function for each node in the network
     */
    public init(shape: Shape, activationFunction: ActivationFunction = .identity) {
        weights = [Matrix](repeating: Matrix(), count: shape.count - 1)
        biases  = [Matrix](repeating: Matrix(), count: shape.count - 1)
        
        for i in 0..<(shape.count - 1) {
            weights[i] = Matrix(rows: shape[i + 1], cols: shape[i])
            biases[i] = Matrix(rows: shape[i], cols: 1)
        }
        
        self.activationFunction = activationFunction
    }
    
    /**
     * Creates a neural network with random weights and biases, with a certain network dimension.
     *
     * - Parameter shape: See the documentation for `init(shape: [Int])`
     * - Parameter shouldIncludeBiases: If `false`, this network will only compute with weights, biases will be all set to zero.
     * - Parameter activationFunction: The activation function for the random network
     * - Parameter weightRange: A constraint for possible values that can be generated as random weights for this network
     * - Parameter biasRange: A constraint for possible values that can be generated as random biases for this network
     */
    public init(randomWithShape shape: Shape, withBiases shouldIncludeBiases: Bool = true, activationFunction: ActivationFunction, weightRange: ClosedRange<Double> = 0...1, biasRange: ClosedRange<Double> = 0...1) {
        
        weights = [Matrix](repeating: Matrix(), count: shape.count - 1)
        biases = [Matrix](repeating: Matrix(), count: shape.count - 1)
        
        
        for i in 0..<weights.count {
            weights[i] = Matrix.random(rows: shape[i + 1], cols: shape[i], range: weightRange)
        }
        
        if shouldIncludeBiases {
            for i in 0..<biases.count {
                biases[i] = Matrix.random(rows: shape[i], cols: 1, range: biasRange)
            }
        }
        
        
        self.activationFunction = activationFunction
    }
    
    /**
     * Decodes raw `NeuralNetwork` data from a buffer.
     *
     * - Parameter buffer: The buffer of data to decode
     */
    init(buffer: UnsafeRawBufferPointer) {
        var readAddress = buffer.baseAddress!
        let layerCount = NeuralNetwork.readInteger(from: &readAddress)
        
        self.activationFunction = ActivationFunction(
            identifier: ActivationFunction.Identifier(
                rawValue: NeuralNetwork.readInteger(from: &readAddress)
            )!
        )
        
        self.weights = [Matrix](repeating: Matrix(), count: layerCount - 1)
        self.biases  = [Matrix](repeating: Matrix(), count: layerCount - 1)
        
        for i in 0..<weights.count {
            weights[i] = Matrix.unsafeRead(from: &readAddress)
        }
        
        for i in 0..<biases.count {
            biases[i] = Matrix.unsafeRead(from: &readAddress)
        }
        
    }
    
    /**
     * Creates a copy of another neural network
     *
     * - Parameter other: Another neural network to copy
     */
    public init(_ other: NeuralNetwork) {
        self.weights = other.weights
        self.biases = other.biases
        self.activationFunction = other.activationFunction
    }
    
    /**
     * Decodes an encoded `NeuralNetwork` from a `Data` object
     *
     * - Parameter data: The encoded neural network to decode
     */
    public convenience init(data: Data) {
        let buffer = data.withUnsafeBytes { $0 }
        self.init(buffer: buffer)
    }
    
    // MARK: Computed Properties
    
    /**
     * This neural network encoded as a `Data` object
     */
    public var encodedData: Data {
        encodeAsData()
    }
    
    // MARK: Methods
    
    /**
     * Encodes this `NeuralNetwork` as a `Data` object
     */
    fileprivate func encodeAsData() -> Data {
        let buffer = write()
        return Data(bytes: buffer.baseAddress!, count: buffer.count)
    }
    
    /**
     * Computes the output layer for a given input layer
     *
     * - Precondition: `input.colCount == 1 && input.count == weights[0].colCount`
     *
     * - Parameter input: A vector representing the activations for the input layer of this matrix
     *
     * - Returns: The vector representing the activations of the output layer after the network feeds forward the input layer.
     */
    public func computeOutputLayer(forInput input: Matrix) -> Matrix {
        var currentLayer = input
        
        for i in 0..<weights.count {
            currentLayer = weights[i] * currentLayer
            currentLayer.add(biases[i])
            currentLayer.applyToAll(activationFunction.apply)
        }
        
        return currentLayer
    }
    
    /**
     * Computes the activations for each layer for some input to this neural network
     *
     * - Parameter input: The input vector
     * - Parameter cache: The list of activations
     *
     * - Precondition: `cache.count == layerCount`
     * - Precondition: `input.colCount == 1 && input.count == weights[0].colCount`
     */
    public func feedForward(input: Matrix, cache: inout [Matrix]) {
        var currentLayer = input
        cache[0] = input
        
        for i in 0..<weights.count {
            currentLayer = weights[i] * currentLayer
            currentLayer.add(biases[i])
            currentLayer.applyToAll(activationFunction.apply)
            cache[i + 1] = currentLayer
        }
    }
    
    /**
     * Computes the activations for each layer for some input to this neural network
     *
     * - Parameter input: The input vector
     * - Parameter cache: The list of activations
     * - Parameter beforeAdjustedCache: The list of input activations BEFORE the activation function is applied
     *
     * - Precondition: `cache.count == layerCount`
     * - Precondition: `beforeAdjustedCache.count == layerCount - 1`
     * - Precondition: `input.colCount == 1 && input.count == weights[0].colCount`
     */
    public func feedForward(input: Matrix, cache: inout [Matrix], beforeAdjustedCache: inout [Matrix]) {
        var currentLayer = input
        cache[0] = input
        
        for i in 0..<weights.count {
            currentLayer = weights[i] * currentLayer
            currentLayer.add(biases[i])
            beforeAdjustedCache[i] = currentLayer
            currentLayer.applyToAll(activationFunction.apply)
            cache[i + 1] = currentLayer
        }
    }
    
    /**
     * Computes the loss for a particular training example
     *
     * - Parameter example: A `DataSet.Item` to compute the loss for
     *
     * - Returns: The sum of the squares of the differences between the expected and computed output activations
     */
    public func cost(for example: DataSet.Item) -> Double {
        computeOutputLayer(forInput: example.input).distanceSquared(from: example.output)
    }
    
    // MARK: Internal Helpers
    
    internal func compute(layer: Int, forInput input: Matrix) -> Matrix {
        var currentLayer = input
        
        for i in 0..<layer {
            currentLayer = weights[i] * currentLayer
            currentLayer.add(biases[i])
            currentLayer.applyToAll(activationFunction.apply)
        }
        
        return currentLayer
    }
    
}
