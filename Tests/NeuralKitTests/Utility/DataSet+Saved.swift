//
//  DataSets.swift
//  
//
//  Created by Sylvan Martin on 7/23/22.
//

import Foundation
import NeuralKit

extension DataSet {
    
    static var mnist = try! DataSet(name: "Digits", inDirectory: URL(fileURLWithPath: "/Users/sylvanm/Programming/Machine Learning/Data sets/NKDS Sets/MNIST Digits"))
    
}
