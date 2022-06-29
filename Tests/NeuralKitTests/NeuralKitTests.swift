import XCTest
@testable import NeuralKit
import MatrixKit

final class NeuralKitTests: XCTestCase {
    
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
