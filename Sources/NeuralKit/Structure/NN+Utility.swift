//
//  NN+Utility.swift
//  
//
//  Created by Sylvan Martin on 7/1/22.
//

import Foundation
import MatrixKit

public extension NeuralNetwork {
    
    // MARK: Shape
    
    /**
     * Reports the shape of this neural network
     */
    var shape: Shape {
        [weights[0].colCount] + biases.map { $0.count }
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
        } + MemoryLayout<Int>.size * 2
        
        let writeBuffer = UnsafeMutableRawBufferPointer.allocate(byteCount: size, alignment: 1)
        
        let writeLayerCountAddress = writeBuffer.bindMemory(to: Int.self).baseAddress!
        writeLayerCountAddress.pointee = weights.count + 1
        
        let actFuncAddress = writeLayerCountAddress.advanced(by: 1)
        actFuncAddress.pointee = self.activationFunction.identifier.rawValue
        
        var matrixAddress = writeBuffer.baseAddress!.advanced(by: 2 * MemoryLayout<Int>.size)
        
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
