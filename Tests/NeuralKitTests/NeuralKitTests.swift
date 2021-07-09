    import XCTest
    import MatrixKit
    @testable import NeuralKit

    final class NeuralKitTests: XCTestCase {
        
        func testLoadingData() {
            // This is an example of a functional test case.
            // Use XCTAssert and related functions to verify your tests produce the correct
            // results.
            
            // For test training purposes, we will do the classes "Hello World" of recognizing handwritten digits
            
            let digitsNN = NeuralNetwork(layerSizes: [784, 16, 16, 10])
            let trainer  = NNTrainer(digitsNN)
            
            // Function that turns a digit into an output layer of the neural network
            func digitToLayer(_ digit: Int) -> Matrix {
                var layer = [Double](repeating: 0, count: 10)
                layer[digit] = 1
                return Matrix(fromCols: [layer])!
            }
            
            let homeDir = FileManager.default.homeDirectoryForCurrentUser
            let trainingDir = homeDir
                .appendingPathComponent("Programming")
                .appendingPathComponent("Frameworks")
                .appendingPathComponent("Swift Packages")
                .appendingPathComponent("NeuralKit")
                .appendingPathComponent("Training Data")
            
            let trainingInputFile   = trainingDir.appendingPathComponent("training_images")
            let trainingOutputFile  = trainingDir.appendingPathComponent("training_labels")
            let testingInputFile    = trainingDir.appendingPathComponent("testing_images")
            let testingOutputFile   = trainingDir.appendingPathComponent("testing_labels")
            
            // open the file and get that training data!
        
            var trainingInputs:  [Matrix] = []
            var trainingOutputs: [Matrix] = []
            
            var testingInputs:  [Matrix] = []
            var testingOutputs: [Matrix] = []
            
            // First load the outputs
            
            
            print("Loading training outputs...", terminator: "")
            trainingOutputs = (try! String(contentsOf: trainingOutputFile).split(whereSeparator: { $0.isNewline })).map { string in
                digitToLayer(Int(string)!)
            }
            print("Done!")
            
            print("Loading testing outputs...", terminator: "")
            testingOutputs = (try! String(contentsOf: testingOutputFile).split(whereSeparator: { $0.isNewline })).map { string in
                digitToLayer(Int(string)!)
            }
            print("Done!")
            
            // Now load the inputs
            
            print("Loading training inputs...", terminator: "")
            trainingInputs = (try! String(contentsOf: trainingInputFile).split(whereSeparator: { $0.isNewline })).map { string in
                Matrix(fromCols: [(try! JSONSerialization.jsonObject(with: string.data(using: .utf8)!, options: .allowFragments) as! [Double]).map { $0 / 255 }])!
            }
            print("Done!")
            
            print("Loading testing inputs...", terminator: "")
            testingInputs = (try! String(contentsOf: testingInputFile).split(whereSeparator: { $0.isNewline })).map { string in
                Matrix(fromCols: [(try! JSONSerialization.jsonObject(with: string.data(using: .utf8)!, options: .allowFragments) as! [Double]).map { $0 / 255 }])!
            }
            print("Done!")
            
            trainer.load(trainingInputs: trainingInputs, trainingOutputs: trainingOutputs)
            trainer.load(trainingInputs: testingInputs, trainingOutputs: testingOutputs, isTestingSet: true)
            
            print("Trainer class has loaded all data")
            
            print("Testing an input")
            
            print(trainer.testingItems[0].input)
            print("Should result in:")
            print(trainer.testingItems[0].output)
            
            print("Actual output from network:")
            print(digitsNN.compute(from: trainer.testingItems[0].input))
            
        }
        
        func testEncoding() {
            
            let network = NeuralNetwork(layerSizes: [5, 5, 5, 5])
            let encoded = network.encodedData()
            
            let homeDir = FileManager.default.homeDirectoryForCurrentUser
            let fileName = homeDir
                .appendingPathComponent("test.nn")
            
            do {
                try encoded.write(to: fileName)
                let readData = try Data(contentsOf: fileName)
                
                let decodedNetwork = NeuralNetwork(fromData: readData)
                
                print(network.biases)
                
                print("Now the other one")
                print(decodedNetwork.biases)
                
            } catch {
                print(error)
            }
            

            
            
        }
        
    }
