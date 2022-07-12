//
//  SGDTests.swift
//  
//
//  Created by Sylvan Martin on 7/12/22.
//

import XCTest
import NeuralKit

class SGDTests: XCTestCase {

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }
    
    func testComputingGradient() throws {
        
        let digitsNetwork = NeuralNetwork(randomWithShape: [784, 16, 16, 10], activationFunction: .sigmoid)
        
        
        
    }

}
