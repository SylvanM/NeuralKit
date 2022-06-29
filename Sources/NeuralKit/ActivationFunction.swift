//
//  ActivationFunction.swift
//  
//
//  Created by Sylvan Martin on 6/28/22.
//

import Foundation

public struct ActivationFunction {
    
    // MARK: Properties
    
    /**
     * The closure used to compute the activation function
     */
    public let compute: (Double) -> Double
    
    /**
     * The derivative of `compute`
     *
     * - Invariant: This is the derivative of `compute(x)` for all `x`
     */
    public let derivative: (Double) -> Double
    
    /**
     * Applies this activation function in place
     */
    public func apply(to x: inout Double) {
        x = compute(x)
    }
    
}

/**
 * A collection of default activation functions
 */
public extension ActivationFunction {
    
    /**
     * The Identity function
     */
    static let identity = ActivationFunction { x in
        x
    } derivative: { _ in
        1
    }

    
    /**
     * The rectified linear unit function
     */
    static let relu = ActivationFunction { x in
        max(0, x)
    } derivative: { x in
        x <= 0 ? 0 : 1
    }
    
    /**
     * The sigmoid function
     */
    static let sigmoid = ActivationFunction { x in
        1 / (1 + exp(-x))
    } derivative: { x in
        (1 / (1 + exp(-x))) - (1 / pow(1 / (1 + exp(-x)), 2))
    }
    
    /**
     * The hyperbolic tangent function
     */
    static let hyperTan = ActivationFunction(compute: tanh) { x in
        1 - pow(tanh(x), 2)
    }
    
    /**
     * The unit step function centered at zero
     */
    static let step = ActivationFunction { x in
        x <= 0 ? 0 : 1
    } derivative: { _ in 0 }

    
}
