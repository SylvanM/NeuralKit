//
//  NN+Utility.swift
//  
//
//  Created by Sylvan Martin on 7/1/22.
//

import Foundation
import MatrixKit

public extension NeuralNetwork {
    
    // MARK: Form Conversion
    
    /**
     * Combines a weight matrix and bias vector into one "weight-bias" matrix
     */
    static func combine(weights: Matrix, biases: Matrix) -> Matrix {
        weights.sideConcatenating(biases).bottomConcatenating(
            Matrix([Matrix.Element](repeating: 0, count: weights.colCount) + [1])
        )
    }
    
    /**
     * Unwraps a weight-bias matrix into a weight matrix and bias vector
     */
    static func unwrap(weightBias: Matrix) -> (weights: Matrix, biases: Matrix) {
        (
            weights: weightBias[0..<(weightBias.rowCount - 1), 0..<(weightBias.colCount - 1)],
            biases: weightBias[0..<(weightBias.rowCount - 1), (weightBias.colCount - 1)..<weightBias.colCount]
        )
    }
    
    // MARK: Shape
    
    /**
     * Reports the shape of this neural network
     */
    var shape: Shape {
        [wbs[0].colCount - 1] + wbs.map { $0.rowCount - 1 }
    }
    
    // MARK: Encoding and Decoding
    
    /**
     * Writes the data of this network to a buffer.
     *
     * - Returns: An `UnsafeRawBufferPointer` consisting of the data of this neural network
     */
    func write() -> UnsafeRawBufferPointer {
        
        // first, write the weights and biases to buffers
        
        let matrices = weights + biases
        let matrixBuffers = matrices.map { $0.encodedDataBuffer }
        
        // size of necessary buffer, in bytes
        let size = matrixBuffers.reduce(into: 0) { partialResult, buffer in
            partialResult += buffer.count
        }
        + MemoryLayout<Int>.size // Writing the layer count
        + MemoryLayout<Int>.size * activationFunctions.count // Write the activation function for each layer
        
        let writeBuffer = UnsafeMutableRawBufferPointer.allocate(byteCount: size, alignment: 1)
        
        let writeLayerCountAddress = writeBuffer.bindMemory(to: Int.self).baseAddress!
        writeLayerCountAddress.pointee = layerCount
        
        let actFuncAddresses = UnsafeMutableBufferPointer<Int>(start: writeLayerCountAddress.advanced(by: 1), count: activationFunctions.count)
        
        for i in 0..<activationFunctions.count {
            actFuncAddresses[i] = activationFunctions[i].identifier.rawValue
        }
        
        var matrixAddress = UnsafeMutableRawPointer(actFuncAddresses.baseAddress!.advanced(by: actFuncAddresses.count))
        
        matrices.forEach { matrix in
            matrix.unsafeWrite(to: &matrixAddress)
        }
        
        return UnsafeRawBufferPointer(writeBuffer)
        
    }
    
    /**
     * Reads an integer an address, returning the next address to read from
     *
     * - Parameter address: An `UnsafeRawPointer` pointing to an encoded `Int`. Upon completion, this value is updated to point
     * to beginning of the next bytes to read
     *
     * - Returns: The value of the underlying bytes as an `Int`
     */
    internal static func readInteger(from address: inout UnsafeRawPointer) -> Int {
        let decoder = address.bindMemory(to: Int.self, capacity: 1)
        let integer = decoder.pointee
        address = UnsafeRawPointer(decoder.advanced(by: 1))
        return integer
    }
 
}
