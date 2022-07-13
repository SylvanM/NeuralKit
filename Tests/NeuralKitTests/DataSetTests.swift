//
//  DataSetTests.swift
//  
//
//  Created by Sylvan Martin on 7/10/22.
//

import XCTest
import NeuralKit

class DataSetTests: XCTestCase {
    
    // MARK: Known Safe URLs
    
    typealias DSAccessInfo = (name: String, url: URL)
    
    /**
     * Information for accessing the known XOR data set
     */
    let xorInfo: DSAccessInfo = (
        name: "XOR",
        url: URL(fileURLWithPath: "/Users/sylvanm/Programming/Machine Learning/Data sets/NKDS Sets/XOR Data Set")
    )
    
    /**
     * Information for accessing the MNIST data set
     */
    let mnistInfo: DSAccessInfo = (
        name: "Digits",
        url: URL(fileURLWithPath: "/Users/sylvanm/Programming/Machine Learning/Data sets/NKDS Sets/MNIST Digits")
    )
    
    let mnistIDXDir = URL(fileURLWithPath: "/Users/sylvanm/Programming/Machine Learning/Data sets/MNIST Digits")
    
    // MARK: Template Code

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    func testExample() throws {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct results.
        // Any test you write for XCTest can be annotated as throws and async.
        // Mark your test throws to produce an unexpected failure when your test encounters an uncaught error.
        // Mark your test async to allow awaiting for asynchronous code to complete. Check the results with assertions afterwards.
    }

    func testPerformanceExample() throws {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }
    
    // MARK: MNIST Data Utility Tests
    
    func convertMNSIT() throws {
        let trainingImages = mnistIDXDir.appendingPathComponent("train-images.idx3-ubyte")
        let trainingLabels = mnistIDXDir.appendingPathComponent("train-labels.idx1-ubyte")
        
        let testingImages = mnistIDXDir.appendingPathComponent("t10k-images.idx3-ubyte")
        let testingLabels = mnistIDXDir.appendingPathComponent("t10k-labels.idx1-ubyte")
        
        let nkdsTrainingURL = mnistInfo.url.appendingPathComponent("Digits_train_data").appendingPathExtension("nkds")
        let nkdsTestingURL = mnistInfo.url.appendingPathComponent("Digits_test_data").appendingPathExtension("nkds")
        
        try MNISTUtility.convert(imageFileURL: trainingImages, labelFileURL: trainingLabels, nkdsFileDes: nkdsTrainingURL)
        try MNISTUtility.convert(imageFileURL: testingImages, labelFileURL: testingLabels, nkdsFileDes: nkdsTestingURL)
    }
    
    func testMNISTData() throws {
        
        let mnistDataSet = try DataSet(name: mnistInfo.name, inDirectory: mnistInfo.url)
        
        var counter = 0
        mnistDataSet.iterateTrainingData { item in
            if counter % 1000 == 0 {
                let mnistItem = MNISTUtility.MNISTItem(item)
                print(mnistItem)
            }
            counter += 1
        }
        
    }
    
    // MARK: Reading Known Data Sets
    
    func testReadingKnownData() throws {
        let xorDataSet = try DataSet(name: xorInfo.name, inDirectory: xorInfo.url)
        
        var count = 0
        xorDataSet.iterateTestingData { item in
            let (a, b) = (Int(item.input[0][0]), Int(item.input[1][0]))
            let saved = Int(item.output[0][0])
            XCTAssertEqual(a ^ b, saved)
            count += 1
        }
        
        XCTAssertEqual(count, 4)
        
        count = 0
        xorDataSet.iterateTrainingData { item in
            let (a, b) = (Int(item.input[0][0]), Int(item.input[1][0]))
            let saved = Int(item.output[0][0])
            XCTAssertEqual(a ^ b, saved)
            count += 1
        }
        
        XCTAssertEqual(count, 4)
    }
    
    // MARK: 

}
