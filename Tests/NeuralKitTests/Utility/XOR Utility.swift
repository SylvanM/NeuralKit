//
//  XOR Utility.swift
//  
//
//  Created by Sylvan Martin on 7/14/22.
//

import Foundation
import NeuralKit

class XORItem: DataSet.Item {
    
    init(_ item: DataSet.Item) {
        super.init(input: item.input, output: item.output)
    }
    
    override var description: String {
        "Training Example: \(Int(input[0])) ^ \(Int(input[1])) = \(Int(output[0, 0]))"
    }
    
}
