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
     * Internal representation of the neural network, using the trick where the biases are represented by just having a constant "one" as the value for the bottom of
     * each layer
     */
    private var parameters: [Matrix]
    
    /**
     * The weight matrices for each layer of the neural network
     */
    public var weights: [Matrix] {
        get {
            parameters.map { parameterMatrix in
                parameterMatrix[rows: 0..<parameterMatrix.rowCount, cols: 0..<(parameterMatrix.rowCount - 1)]
            }
        } set {
            for i in 0..<newValue.count {
                parameters[i][rows: 0..<newValue[i].rowCount, cols: 0..<newValue[i].colCount] = newValue[i]
            }
        }
    }
    
    /**
     * The biases of each layer of the neural network
     */
    public var biases: [Matrix] {
        get {
            parameters.map { parameterMatrix in
                parameterMatrix[rows: 0..<parameterMatrix.rowCount, cols: (parameterMatrix.colCount - 1)..<parameterMatrix.colCount]
            }
        } set {
            for i in 0..<newValue.count {
                parameters[i][rows: 0..<newValue[i].rowCount, cols: (newValue[i].colCount - 1)..<newValue[i].colCount] = newValue[i]
            }
        }
    }
    
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
        if biases.count != weights.count {
            print("Unequal amount of weights and biases")
            return false
        }
        
        for i in 0..<(weights.count - 1) {
            if weights[i + 1].colCount != weights[i].rowCount {
                print("Invalid weight dimensions")
                return false
            }
        }
        
        for i in 0..<biases.count {
            if biases[i].colCount != 1 {
                print("Biases aren't vectors")
                return false
            }
            if biases[i].rowCount != weights[i].rowCount {
                print("Invalid bias dimension: expected \(weights[i].rowCount) but got \(biases[i].rowCount)")
                return false
            }
        }
        
        return true
    }
    
    /**
     * The activation functions for layers of this neural network.
     *
     * This function is the final step in the computation of the activations for a layer, after the weights and biases have been applied.
     *
     * - Invariant: `activationFunctions.count == biases.count`
     */
    public var activationFunctions: [ActivationFunction]
    
    // MARK: Initializers
    
    /**
     * Creates a network from certain weights and biases
     *
     * - Precondition: These satisfy the invariant
     *
     * - Parameter weights: The weights for the neural network
     * - Parameter biases: The biases for each activation layer other than the input layer.
     * - Parameter activationFunctions: The activation functions for this neural network.
     */
    public init(weights: [Matrix], biases: [Matrix], activationFunctions: [ActivationFunction]) {
       
        self.parameters = zip(weights, biases).map { (weightMatrix, biasVector) in
            weightMatrix.sideConcatenating(biasVector)
        }
        
        self.activationFunctions = activationFunctions
    }
    
    /**
     * Creates an empty neural network of a specific shape
     *
     * - Parameter shape: An array describing the size of each layer. If `shape` is `[2, 2, 1]`, a neural network will be created with
     * 2 input nodes, one hidden layer with 2 nodes, and an output layer with 1 node.
     * - Parameter activationFunctions: The activation functions for each layer in the network
     */
    public init(shape: Shape, activationFunctions: [ActivationFunction]) {
        
        
        
        parameters = [Matrix](repeating: Matrix(), count: shape.count - 1)
        
        for i in 0..<(shape.count - 1) {
            parameters[i] = Matrix(rows: shape[i + 1], cols: shape[i]).sideConcatenating(Matrix(rows: shape[i + 1], cols: 1))
        }
        
        self.activationFunctions = activationFunctions
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
    public convenience init(randomWithShape shape: Shape, activationFunctions: [ActivationFunction], weightRange: ClosedRange<Double> = -5...5, biasRange: ClosedRange<Double> = -5...5) {
        self.init(shape: shape, activationFunctions: activationFunctions)
        for p in 0..<parameters.count {
            for r in 0..<weights[p].rowCount {
                biases[p][r] = Double.random(in: biasRange)
                for c in 0..<weights[p].colCount {
                    weights[p][r, c] = Double.random(in: weightRange)
                }
            }
        }
    }
    
    /**
     * Decodes raw `NeuralNetwork` data from a buffer.
     *
     * - Parameter buffer: The buffer of data to decode
     */
    convenience init(buffer: UnsafeRawBufferPointer) {
        var readAddress = buffer.baseAddress!
        let layerCount = NeuralNetwork.readInteger(from: &readAddress)
        
        var activationFunctions = [ActivationFunction](repeating: .identity, count: layerCount - 1)
        
        for i in 0..<activationFunctions.count {
            activationFunctions[i] = ActivationFunction(
                identifier: ActivationFunction.Identifier(
                    rawValue: NeuralNetwork.readInteger(from: &readAddress)
                )!
            )
        }
        
        var weights = [Matrix](repeating: Matrix(), count: layerCount - 1)
        var biases  = [Matrix](repeating: Matrix(), count: layerCount - 1)
        
        for i in 0..<weights.count {
            weights[i] = Matrix.unsafeRead(from: &readAddress)
        }
        
        for i in 0..<biases.count {
            biases[i] = Matrix.unsafeRead(from: &readAddress)
        }
        
        self.init(weights: weights, biases: biases, activationFunctions: activationFunctions)
        
    }
    
    /**
     * Creates a copy of another neural network
     *
     * - Parameter other: Another neural network to copy
     */
    public init(_ other: NeuralNetwork) {
        self.parameters = other.parameters
        self.activationFunctions = other.activationFunctions
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
            currentLayer.applyToAll(activationFunctions[i].apply)
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
            currentLayer.applyToAll(activationFunctions[i].apply)
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
            currentLayer.applyToAll(activationFunctions[i].apply)
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
            currentLayer.applyToAll(activationFunctions[i].apply)
        }
        
        return currentLayer
    }
    
}
