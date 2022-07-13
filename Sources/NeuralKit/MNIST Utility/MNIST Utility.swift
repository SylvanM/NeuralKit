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
 * A type that iterates through an IDX file and returns the next represented vector when requested.
 */
public protocol IDXIterator {
    
    /**
     * The number of total examples in the entire file
     */
    var count: Int { get }
    
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
 * A class containing methods for converting from the MNIST IDX files to NeuralKit NKDS files
 */
public class MNISTUtility {
    
    public class MNISTItem: DataSet.Item {
        
        public init(_ item: DataSet.Item) {
            super.init(input: item.input, output: item.output)
        }
        
        public override var description: String {
            
            var digit = 0
            
            for i in 0..<output.flatmap.count {
                if output.flatmap[i] == 1 {
                    digit = i
                    break
                }
            }
            
            var desc = "Handwitten digit: \(digit)"
            
            let pixels = input.flatmap.map { brightness -> String in
                if brightness <= 0.2 {
                    return " "
                } else if brightness <= 0.4 {
                    return "░"
                } else if brightness <= 0.6 {
                    return "▒"
                } else if brightness <= 0.8 {
                    return "▓"
                } else {
                    return "█"
                }
            }

            for i in 0..<(28 * 28) {
                if i % 28 == 0 {
                    desc += "\n"
                }
                
                desc += pixels[i]
            }
            
            return desc
        }
        
    }
    
    /**
     * A class for iterating through the IDX MNIST image file
     */
    class IDXPixelIterator: IDXIterator {
        
        var handle: FileHandle
        
        var count: Int
        
        /**
         * Sets up the iterator for reading each example of a handwritten digit
         */
        required init(forFile url: URL) throws {
            self.handle = try FileHandle(forReadingFrom: url)
            self.count = 0
            
            // skip the first 4 bytes, we are ASSUMING their values.
            _ = try handle.read(upToCount: 4)
            
            guard let countData = try? handle.read(upToCount: 4) else {
                throw DataSet.DSFileError.cannotReadIDXFileData
            }
            
            self.count = countData.withUnsafeBytes { buffer in
                Int(buffer.bindMemory(to: Int32.self).baseAddress!.pointee.bigEndian)
            }
            
            _ = try handle.read(upToCount: 8)
        }
        
        /**
         * Returns a vector representing the pixels of a handwritten image
         */
        func next() -> Matrix? {
            guard let nextData = try? handle.read(upToCount: 28 * 28) else { return nil }
            
            return nextData.withUnsafeBytes { buffer in
                let pixels = buffer.bindMemory(to: UInt8.self)
                let flatmap = pixels.map { Double($0) / Double(UInt8.max) }
                return Matrix(vector: flatmap)
            }
        }
        
    }
    
    /**
     * A class for iterating through the IDX MNIST label file
     */
    class IDXLabelIterator: IDXIterator {
        
        var handle: FileHandle
        
        var count: Int
        
        required init(forFile url: URL) throws {
            self.handle = try FileHandle(forReadingFrom: url)
            self.count = 0
            
            // skip the first 4 bytes, we are ASSUMING their values.
            _ = try handle.read(upToCount: 4)
            
            guard let countData = try? handle.read(upToCount: 4) else {
                throw DataSet.DSFileError.cannotReadIDXFileData
            }
            
            self.count = countData.withUnsafeBytes { buffer in
                Int(buffer.bindMemory(to: Int32.self).baseAddress!.pointee.bigEndian)
            }
            
        }
        
        func next() -> Matrix? {
            guard let nextData = try? handle.read(upToCount: 1) else { return nil }
            
            return nextData.withUnsafeBytes { buffer in
                let label = buffer.bindMemory(to: UInt8.self).baseAddress!.pointee
                var flatmap = [Double](repeating: 0, count: 10)
                flatmap[Int(label)] = 1
                return Matrix(vector: flatmap)
            }
        }
        
    }
    
    // MARK: Methods
    
    /**
     * Converts an MNIST IDX data set into one NKDS file
     *
     * - Parameter imageFileURL: The URL of the file containing image pixel data
     * - Parameter labelFileURL: The URL of the file containing the labels for each image
     * - Parameter nkdsFileDes: The output file to write to
     */
    public static func convert(imageFileURL: URL, labelFileURL: URL, nkdsFileDes: URL) throws {
        let imageIterator = try IDXPixelIterator(forFile: imageFileURL)
        let labelIterator = try IDXLabelIterator(forFile: labelFileURL)
        
        let outHandle = try FileHandle(forWritingTo: nkdsFileDes)
        
        let firstImage = imageIterator.next()!
        let firstLabel = labelIterator.next()!
        
        let firstItem = DataSet.Item(input: firstImage, output: firstLabel)
        
        let size = firstItem.toBuffer().count
        
        try DataSet.write(count: imageIterator.count, size: size, toHandle: outHandle)
        try DataSet.write(item: firstItem, to: outHandle)
        
        while let labelVector = labelIterator.next() {
            let imageVector = imageIterator.next()!
            let item = DataSet.Item(input: imageVector, output: labelVector)
            try DataSet.write(item: item, to: outHandle)
        }
        
        try outHandle.close()
    }
    
}
