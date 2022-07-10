//
//  DataSet.swift
//  
//
//  Created by Sylvan Martin on 7/5/22.
//

import Foundation
import MatrixKit

/**
 * A class managing reading and writing data sets to and from, using a custom data format.
 *
 * I'm calling the file format "NeuralKit Data Set", abbreviated as NKDS, so all data set files will have the
 * `.nkds` extension.
 *
 * NKDS Order is
 *
 * - 8 byte prefix that contains an integer descriping how many items are in the file
 * - 8 bytes containing the size (in bytes) of each item
 * - The items
 */
public class DataSet {
    
    // MARK: Properties
    
    /**
     * The total number of training items
     */
    public fileprivate(set) var trainingItemsCount: Int
    
    /**
     * The total number of testing items
     */
    public fileprivate(set) var testingItemsCount: Int
    
    /**
     * The name of this data set
     */
    public let name: String
    
    /**
     * The directory this data set is stored in, containing the training data and testing data.
     *
     * - Invariant: Not sure if this is technically an invariant, but the files in the directory should consist of
     *  - `(name)_train_data.nkds`
     *  - `(name)_test_data.nkds`
     */
    public let directory: URL
    
    /**
     * The file path for the training items
     */
    private var trainingFileURL: URL {
        directory.appendingPathComponent("\(name)" + DataSet.trainingFilePostfix).appendingPathExtension(DataSet.fileExtension)
    }
    
    /**
     * The file path for testing items
     */
    private var testingFileURL: URL {
        directory.appendingPathComponent("\(name)" + DataSet.testingFilePostfix).appendingPathExtension(DataSet.fileExtension)
    }
    
    // MARK: Static Properties
    
    private static let fileExtension = "nkds"
    private static let trainingFilePostfix = "_train_data"
    private static let testingFilePostfix = "_test_data"
    
    // MARK: Structures
    
    /**
     * A single training item
     */
    public struct Item {
        
        /**
         * The input vector
         */
        public let input: Matrix
        
        /**
         * The expected output vector
         */
        public let output: Matrix
        
        /**
         * Creates a training item from an input and output
         */
        public init(input: Matrix, output: Matrix) {
            self.input = input
            self.output = output
        }
        
    }
    
    // MARK: Initializers
    
    /**
     * Creates a dataset from an array of training examples and writes them to a file
     *
     * - Parameter name: The name of this data set
     * - Parameter training
     */
    public init(name: String, trainingItems: [Item], testingItems: [Item], inDirectory url: URL) throws {
        self.directory = url
        self.name = name
        self.trainingItemsCount = trainingItems.count
        self.testingItemsCount = testingItems.count
        
        try DataSet.write(items: trainingItems, to: trainingFileURL)
        try DataSet.write(items: testingItems, to: testingFileURL)
    }
    
    /**
     * Creates a `DataSet` object from a directory already containing data
     *
     * - Parameter name: The name of this data set
     * - Parameter url: The URL of the directory containing the training and testing data
     */
    public init(name: String, inDirectory url: URL) throws {
        self.name = name
        self.directory = url
        
        // this is only so we can use the self computed properties later. this will be updated.
        self.trainingItemsCount = 0
        self.testingItemsCount = 0
        
        let trainingHandle = try FileHandle(forReadingFrom: trainingFileURL)
        let testingHandle = try FileHandle(forReadingFrom: testingFileURL)
        
        // read the counts in each file
        self.trainingItemsCount = try DataSet.readCount(from: trainingHandle)
        self.testingItemsCount = try DataSet.readCount(from: testingHandle)
    }
    
    // MARK: Data Iteration
    
    /**
     * Iterates over all testing data, this is not guarnateed to work in order, and may operate concurrently.
     *
     * - Parameter closure: A closure of how to handle each testing item
     */
    public func iterateTestingData(_ closure: (Item) -> ()) {
        let testingFile = try! FileHandle(forReadingFrom: testingFileURL)
        try! iterate(fileHandle: testingFile, closure)
    }
    
    /**
     * Iterates over all training data, this is not guarnateed to work in order, and may operate concurrently.
     *
     * - Parameter closure: A closure of how to handle each training item
     */
    public func iterateTrainingData(_ closure: (Item) -> ()) {
        let trainingFile = try! FileHandle(forReadingFrom: trainingFileURL)
        try! iterate(fileHandle: trainingFile, closure)
    }
    
    private func iterate(fileHandle: FileHandle, _ closure: (Item) -> ()) throws {
        
        // TODO: Eventually improve this with concurrency, prefetching, and other fun tricks!
        
        let caseAmount = try! fileHandle.read(upToCount: MemoryLayout<Int>.size)!.withUnsafeBytes{
            $0.bindMemory(to: Int.self).baseAddress!.pointee
        }
        
        let caseSize = try! DataSet.readCaseSize(from: fileHandle)
        
        for i in 0..<caseAmount {
            let data = try! fileHandle.read(upToCount: caseSize)!
            data.withUnsafeBytes { buffer in
                let item = Item(unsafeBuffer: buffer)
                closure(item)
            }
        }
        
    }
    
    // MARK: Cost Computation
    
    /**
     * Computes the average cost of a neural network performance for all training items in the dataset
     *
     * - Parameter network: The network to compute the average cost of
     *
     * - Returns: The average cost for each testing item
     */
    public func trainingCost(of network: NeuralNetwork) -> Double {
        averageCost(of: network, iterating: iterateTrainingData, count: Double(trainingItemsCount))
    }
    
    /**
     * Computes the average cost of neural network performance over all testing items
     *
     * - Parameter network: The network to compute the average testing cost of
     *
     * - Returns: The average cost for each testing item.
     */
    public func testingCost(of network: NeuralNetwork) -> Double {
        averageCost(of: network, iterating: iterateTestingData, count: Double(testingItemsCount))
    }
    
    private func averageCost(of network: NeuralNetwork, iterating: ((Item) -> ()) -> (), count: Double) -> Double {
        var average: Double = 0
        
        iterating { item in
            let cost = network.cost(for: item)
            average += cost / count
        }
        
        return average
    }
    
}
