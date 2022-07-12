//
//  MNIST Utility.swift
//  
//
//  Created by Sylvan Martin on 7/11/22.
//

import Foundation
import NeuralKit
import MatrixKit

/**
 * A class containing methods for converting from the MNIST IDX files to NeuralKit NKDS files
 */
class MNISTUtility {
    
    /**
     * A type that iterates through an IDX file and returns the next represented vector when requested.
     */
    protocol IDXIterator {
        
        /**
         * Creates an iterator for a file
         */
        init(forFile: URL) throws
        
        /**
         * Returns the next desired vector from this file
         */
        func next() -> Matrix?
        
    }
    
    /**
     * A class for iterating through the IDX image file
     */
    class IDXPixelIterator: IDXIterator {
        
    }
    
}
