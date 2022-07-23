//
//  XORTests.swift
//  
//
//  Created by Sylvan Martin on 7/23/22.
//

import XCTest
import MatrixKit
import NeuralKit

class XORTests: XCTestCase {

    func testXORFeedForward() throws {
        let w1: Matrix = [
            [1, 1],
            [-1, -1]
        ]
        let b1 = Matrix(vector: [-0.5, 1.5])
        let w2 = Matrix([1, 1])
        let b2 = Matrix([-1.5])
        
        let xorNetwork = NeuralNetwork(weights: [w1, w2], biases: [b1, b2], activationFunctions: [.step, .step])
        
        let inputSpace = [0, 1]
        
        inputSpace.forEach { a in
            inputSpace.forEach { b in
                let outputVector = xorNetwork.computeOutputLayer(forInput: Matrix(vector: [Double(a), Double(b)]))
                let computed = Int(outputVector[0, 0])
                XCTAssertEqual(a ^ b, computed)
            }
        }
    }

}
