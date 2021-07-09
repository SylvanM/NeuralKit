//
//  NN+Encoding.swift
//
//  File containing functions to write and save the NN
//
//  Created by Sylvan Martin on 7/9/21.
//

import Foundation

/**
 * Extension containing functions to read and write this network
 */
@available(macOS 10.12, *)
extension NeuralNetwork {
    
    /**
     * Returns a `Data` object representing the structure of this neural network
     */
    func encodedData() -> Data {
        
        var bytesToEncode: [UInt8] = []
        
        let intSize = MemoryLayout<Int>.size
        
        bytesToEncode.append(UInt8(intSize))
        
        // We need to save the dimension of the neural net. Since the dimension isn't actually stored
        // with the NN object, we need to calculate it by looking at the dimensions of the weights and biases.
        
        var networkDimension = biases.map { $0.rows }
        networkDimension.insert(weights[0].cols, at: 0)
        
        var layers = self.layers
        
        // first, save the number of layers, so that we know how much of the next stuff to decode.
        bytesToEncode.append(contentsOf: withUnsafeBytes(of: &layers, { $0.bindMemory(to: UInt8.self) }))
        
        // save the network dimension as an array of bytes
        bytesToEncode.append(contentsOf: networkDimension.withUnsafeBytes {
            $0.bindMemory(to: UInt8.self)
        })
        
        // now save the weights
        bytesToEncode.append(contentsOf: weights.map { matrix in
            matrix.flatmap.withUnsafeBytes { $0.bindMemory(to: UInt8.self) }
        }.flatMap { $0 })
        
        // now save the biases
        bytesToEncode.append(contentsOf: biases.map { matrix in
            matrix.flatmap.withUnsafeBytes { $0.bindMemory(to: UInt8.self) }
        }.flatMap { $0 })
        
        return Data(bytesToEncode)
        
    }
    
}
