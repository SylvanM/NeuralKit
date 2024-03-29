import XCTest
@testable import NeuralKit
import MatrixKit

final class NeuralKitTests: XCTestCase {
    
    // MARK: Test Utility
    
    func testReadWrite() {
        
        let shape = [2, 2, 1]
        let actfuncs: [ActivationFunction] = [.sigmoid, .sigmoid]
        let net = NeuralNetwork.init(randomWithShape: Array(shape), activationFunctions: actfuncs)
        
        XCTAssertEqual(net.shape, shape)
        
        let encodedNet = net.encodedData
        
        let decodedNet = NeuralNetwork(data: encodedNet)

        XCTAssertEqual(net.weights, decodedNet.weights)
        XCTAssertEqual(net.biases, decodedNet.biases)
        XCTAssertEqual(net.activationFunctions, decodedNet.activationFunctions)
        
        // now try with a bunch of random ones
        
        // this takes really really long, let's not run it unless we are explicitly testing it...
//        for _ in 1...100 {
//            var randomShape = [Int](repeating: 0, count: Int.random(in: 2...100))
//
//            for i in 0..<randomShape.count {
//                randomShape[i] = Int.random(in: 1...1000)
//            }
//
//            let net = NeuralNetwork(randomWithShape: randomShape, activationFunction: .hyperTan)
//
//            let encodedNet = net.encodedData
//            let decodedNet = NeuralNetwork(data: encodedNet)
//
//            XCTAssertEqual(net.weights, decodedNet.weights)
//            XCTAssertEqual(net.biases, decodedNet.biases)
//            XCTAssertEqual(net.activationFunction, decodedNet.activationFunction)
//        }
        
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
            activationFunctions: [.step, .step]
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
     
    // MARK: Test Feed Forward
    
    func testFeedForward() throws {
        let mnistData = try DataSet(name: "Digits", inDirectory: URL(fileURLWithPath: "/Users/sylvanm/Programming/Machine Learning/Data sets/NKDS Sets/MNIST Digits"))
        let digitsNetwork = NeuralNetwork(randomWithShape: [784, 16, 16, 10], withBiases: false, activationFunctions: [.relu, .relu, .sigmoid])
        
        mnistData.iterateTrainingData { item in
            let mnistItem = MNISTUtility.MNISTItem(item)
            var activations = [Matrix](repeating: Matrix(), count: digitsNetwork.layerCount)
            digitsNetwork.feedForward(input: mnistItem.input, cache: &activations)
            
            activations.forEach { activationVector in
                activationVector.forEach {
                    XCTAssertFalse($0.isNaN)
                    XCTAssertFalse($0.isInfinite)
                }
            }
            
            _ = digitsNetwork.cost(for: mnistItem)
        }
    }
    
    func testActivationFunctions() throws {
        let afs: [ActivationFunction] = [.identity, .sigmoid, .relu, .hyperTan, .step]
    
        for af in afs {
            for _ in 0..<10000 {
                let input: Double = .random(in: -10000...10000)
                let output = af.compute(input)
                let der = af.derivative(input)
                
                XCTAssertFalse(output.isNaN)
                XCTAssertFalse(der.isNaN)
            }
        }
        
        let act = ActivationFunction.sigmoid
        let range: ClosedRange<Double> = -1...1
        
    }
    
}
