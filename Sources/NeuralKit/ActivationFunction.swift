//
//  ActivationFunction.swift
//  
//
//  Created by Sylvan Martin on 6/28/22.
//

import Foundation

/**
 * A function that maps the activation of a neuron to a more desirable domain
 *
 * In this library, my philosophy is usually to give as much control to the client as possible while providing useful tools. This is the
 * one exception where there are only static, prewritten functions available. The reason for this is only because of how I am implementing
 * `NeuralNetwork` encoding. If a client could write their own activation function (which I would orefer they be able to do)
 * there would be no easy way of encoding that function so that the network could be saved or written to a file.
 *
 * I really want to allow flexibility on this functionality, so if you can think of a good way to be able to save the user's custom function
 * to a file, please let me know!
 */
public struct ActivationFunction: Equatable {
    
    // MARK: Properties
    
    /**
     * Which activation function this is
     */
    internal let identifier: Identifier
    
    /**
     * The function used to compute activation from a raw neuron value
     */
    public var compute: (Double) -> Double {
        AFGroup.activationFunctions[identifier.rawValue].compute
    }
    
    /**
     * The derivative of `compute`
     *
     * - Invariant: This is the derivative of `compute(x)` for all `x`
     */
    public var derivative: (Double) -> Double {
        AFGroup.activationFunctions[identifier.rawValue].derivative
    }
    
    /**
     * Applies this activation function, in place
     *
     * - Parameter x: The value to apply this activation function to
     */
    public func apply(to x: inout Double) {
        x = compute(x)
    }
    
    /**
     * Applies this activation function's derivative, in place
     *
     * - Parameter x: The value to apply this activation function to
     */
    public func applyDerivative(to x: inout Double) {
        x = derivative(x)
    }
    
    // MARK: Enumerations
    
    /**
     * A type to refer to an activation function
     */
    internal enum Identifier: Int {
        case identity = 0
        case relu = 1
        case sigmoid = 2
        case hyperTan = 3
        case step = 4
    }
    
    // MARK: Initializers
    
    /**
     * To allow for the encoding and decoding of activation functions, I am allowing only a certain number of these functions that are pre-written.
     * To store an activation function, we only care about an identifier attached to each possible function.
     */
    internal init(identifier: Identifier) {
        self.identifier = identifier
    }
    
    /**
     * Creates an activation function from another activation function
     *
     * - Parameter other: An activation function to copy
     */
    public init(other: ActivationFunction) {
        self.identifier = other.identifier
    }
    
}

public extension ActivationFunction {
    
    // MARK: Activation Functions
    
    /**
     * Returns its input.
     */
    static let identity = ActivationFunction(identifier: .identity)
    
    /**
     * For an input `x`, returns `max(0, x)`
     */
    static let relu = ActivationFunction(identifier: .relu)
    
    /**
     * For an input `x`, returns `1 / (1 + exp(-x))`
     */
    static let sigmoid = ActivationFunction(identifier: .sigmoid)
    
    /**
     * For an input `x`, returns the hyperbolic tangent of `x`
     */
    static let hyperTan = ActivationFunction(identifier: .hyperTan)
    
    /**
     * For an input `x`, returns `x <= 0 ? 0 : 1`
     */
    static let step = ActivationFunction(identifier: .step)
    
    
}

/**
 * A collection of an activation function and its derivative
 */
fileprivate struct AFGroup {
    
    /**
     * - Invariant: `derivative` is the derivative of `compute`
     */
    
    let compute: (Double) -> Double
    let derivative: (Double) -> Double
}

// MARK: - Activation Function Definitions

extension AFGroup {
    
    static let identity = AFGroup { x in
        x
    } derivative: { _ in
        1
    }

    static let relu = AFGroup { x in
        max(0, x)
    } derivative: { x in
        x <= 0 ? 0 : 1
    }
    
    fileprivate static func _sig(_ x: Double) -> Double {
        1 / (1 + exp(-x))
    }
    
    fileprivate static func _sigder(_ x: Double) -> Double {
        _sig(x) * (1 - _sig(x))
    }

    static let sigmoid = AFGroup { x in
        _sig(x)
    } derivative: { x in
        _sigder(x)
    }

    static let hyperTan = AFGroup(compute: tanh) { x in
        1 - pow(tanh(x), 2)
    }

    static let step = AFGroup { x in
        x <= 0 ? 0 : 1
    } derivative: { _ in 0 }

    /**
     * The collection of all activation functions
     */
    static let activationFunctions: [AFGroup] = [
        identity, relu, sigmoid, hyperTan, step
    ]

}
