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
    
    /**
     * Information for accessing the known XOR data set
     */
    let xorInfo = (
        name: "XOR",
        url: URL(fileURLWithPath: "/Users/sylvanm/Programming/Machine Learning/Data sets/NKDS Sets/XOR Data Set")
    )
    
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
    
    // MARK: Reading Known Data Sets
    
    func testReadingKnownData() throws {
        let xorDataSet = try DataSet(name: xorInfo.name, inDirectory: xorInfo.url)
    }

}
