//
//  Compression Functions.swift
//  
//
//  Created by Sylvan Martin on 6/29/21.
//

/**
 * This file contains definitions for different functions that can be used to condense the real number line
 * into any other set for the purposes of calculating neuron activation
 */

import Foundation

// delete dis

public struct ActivationFunction {
    
    public var function: (Double) -> Double
    
    public func callAsFunction(_ x: Double) -> Double {
        function(x)
    }
    
    
    
}

extension ActivationFunction {
    
    static let sigmoid = ActivationFunction { x in
        1.0 / (1.0 + exp(-x))
    }
    
    static let absoluteValue = ActivationFunction { x in
        abs(x)
    }
    
    static let identity = ActivationFunction { x in
        x
    }
    
    static let rectifiedLinear = ActivationFunction { x in
        max(0, x)
    }
    
}
