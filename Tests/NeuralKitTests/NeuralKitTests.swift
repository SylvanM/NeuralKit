import XCTest
@testable import NeuralKit
import MatrixKit

final class NeuralKitTests: XCTestCase {
    
    // MARK: Test Utility
    
    func testReadWrite() {
        
        let shape = [2, 2, 1]
        let actfunc = ActivationFunction.hyperTan
        let net = NeuralNetwork.init(randomWithShape: Array(shape), activationFunction: actfunc)
        
        let encodedNet = net.encodedData
        
        // make sure the activation function is encoded right
        encodedNet.withUnsafeBytes { buffer in
            let actCode = buffer.bindMemory(to: Int.self).baseAddress!.pointee
            XCTAssertEqual(actCode, actfunc.identifier.rawValue)
        }
        
        let decodedNet = NeuralNetwork(data: encodedNet)

        XCTAssertEqual(net.weights, decodedNet.weights)
        XCTAssertEqual(net.biases, decodedNet.biases)
        XCTAssertEqual(net.activationFunction, decodedNet.activationFunction)
        
        // now try with a bunch of random ones
        
        // this takes really really long, let's not run it unless we are explicitly testing it...
        for _ in 1...100 {
            var randomShape = [Int](repeating: 0, count: Int.random(in: 2...100))

            for i in 0..<randomShape.count {
                randomShape[i] = Int.random(in: 1...1000)
            }

            let net = NeuralNetwork(randomWithShape: randomShape, activationFunction: .hyperTan)

            let encodedNet = net.encodedData
            let decodedNet = NeuralNetwork(data: encodedNet)

            XCTAssertEqual(net.weights, decodedNet.weights)
            XCTAssertEqual(net.biases, decodedNet.biases)
            XCTAssertEqual(net.activationFunction, decodedNet.activationFunction)
        }
        
    }
    
    // MARK: Basic Structure Tests
    
    func testSimpleXOR() {
        
        let hWeights: Matrix = [
            [1, 1],
            [-1, -1]
        ]
        
        let hBiases = Matrix(vector: [-0.5, 1.5])
        
        let yWeights = Matrix([1, 1])
        
        let yBiases = Matrix([-1.5])
        
        let simpleNetwork = NeuralNetwork(
            weights: [hWeights, yWeights],
            biases: [hBiases, yBiases],
            activationFunction: .step
        )
        
        XCTAssert(simpleNetwork.invariantSatisied)
        
        let inputSpace = [0, 1]
        
        let classifier = Classifier(fromNetwork: simpleNetwork) { outVect -> Int in
            Int(outVect[0, 0])
        }
        
        inputSpace.forEach { x in
            inputSpace.forEach { y in
                let xor = x ^ y
                let computed = classifier.classify([Double(x), Double(y)])
                
                XCTAssertEqual(xor, computed)
            }
        }
        
    }
    
}
