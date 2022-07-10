//
//  DataSet+FileUtility.swift
//  
//
//  Created by Sylvan Martin on 7/8/22.
//

import Foundation
import XCTest
import MatrixKit

extension DataSet {
    
    // MARK: File Utility
    
    /**
     * Converts an IDX formatted file into an NKDS file.
     *
     * - Parameter idxFile: The filepath to the IDX file to convert
     * - Parameter nkdsFileDes: The destination to write the conversion to
     */
    public static func convert(idkFile: URL, writingTo nkdsFileDes: URL) {
        
    }
    
    // MARK: Reading and Writing
    
    /**
     * Writes an array of training items to a particular file
     *
     * - Parameter items: An array (`[DataSet.Item]`) to write to a file
     * - Parameter url: The url of the file to write to
     */
    public static func write(items: [DataSet.Item], to url: URL) throws {
        
        var handle: FileHandle
        
        do {
            handle = try FileHandle(forWritingTo: url)
        } catch CocoaError.fileNoSuchFile {
            // the file doesn't exist, so create it
            FileManager.default.createFile(atPath: url.path, contents: nil)
            handle = try FileHandle(forWritingTo: url)
        }
        
        
        // prepend the files with the number of training items in each one
        let countBuffer = UnsafeMutableBufferPointer<Int>.allocate(capacity: 1)
        
        countBuffer.baseAddress!.pointee = items.count
        
        try handle.write(contentsOf: UnsafeRawBufferPointer(countBuffer))
        
        // now write the size
        let caseSize = items[0].toBuffer().count
        try writeInt(caseSize, to: handle)
        
        // write each item to the file!
        for item in items {
            try handle.write(contentsOf: item.toBuffer())
        }
        
        try handle.close()
    }
    
    private static func writeInt(_ value: Int, to handle: FileHandle) throws {
        let valPtr = UnsafeMutableBufferPointer<Int>.allocate(capacity: 1)
        valPtr.baseAddress!.pointee = value
        try handle.write(contentsOf: UnsafeRawBufferPointer(valPtr))
    }
    
    private static func readInt(from handle: FileHandle) throws -> Int {
        try handle.read(upToCount: MemoryLayout<Int>.size)!.withUnsafeBytes { buffer in
            return buffer.bindMemory(to: Int.self).baseAddress!.pointee
        }
    }
    
    /**
     * Reads the first several bytes of a NKDS file to retrieve the amount of cases
     *
     * - Parameter handle: The handle of the NKDS file
     *
     * - Precondition: `url` refers to a true NKDS file, not some other random file.
     *
     * I'll document that better in the future.
     *
     * - Returns: The number of cases encoded in the file
     */
    public static func readCount(from handle: FileHandle) throws -> Int {
        try readInt(from: handle)
    }
    
    /**
     * Reads the size, in bytes, of each case in a file, and advances the file handle
     *
     * - Parameter handle: A file handle to the file containing learning cases
     *
     * - Precondition: The handle is in the proper location in the file, having just read the number of cases.
     *
     * - Returns: The size, in bytes, of each training case.
     */
    public static func readCaseSize(from handle: FileHandle) throws -> Int {
        try readInt(from: handle)
    }
    
}

extension DataSet.Item {
    
    // MARK: Initializers
    
    /**
     * Creates a `DataSet.Item` from a buffer of encoded data
     *
     * - Precondition: `buffer` points to a properly encoded sequence of bytes of the same format created by `toBuffer()` or `unsafeWrite(to:)`
     *
     * - Parameter buffer: A pointer to the buffer of encoded data
     */
    public init(unsafeBuffer: UnsafeRawBufferPointer) {
        var baseAddress = unsafeBuffer.baseAddress!
        self.init(fromUnsafeBaseAddress: &baseAddress)
    }
    
    /**
     * Creates a training example from the base address of a buffer of data that encodes this training item, and updates the base address to point to the next
     * byte after this buffer
     *
     * - Parameter baseAddress: `UnsafeRawPointer` pointing to the beginning of a byte buffer that encodes a training item, which
     * will be incremented.
     */
    public init(fromUnsafeBaseAddress baseAddress: inout UnsafeRawPointer) {
        input = Matrix.unsafeRead(from: &baseAddress)
        output = Matrix.unsafeRead(from: &baseAddress)
    }
    
    // MARK: Encoding/Decoding
    
    /**
     * Writes the underlying data of this training example to a buffer and returns a pointer to the buffer
     *
     * - Returns: A buffer pointer to the underlying data to this training item
     */
    public func toBuffer() -> UnsafeRawBufferPointer {
        let inputBuffer = input.encodedDataBuffer
        let outputBuffer = output.encodedDataBuffer
        
        let finalBuffer = UnsafeMutableRawBufferPointer.allocate(byteCount: inputBuffer.count + outputBuffer.count, alignment: 1)
        inputBuffer.copyBytes(to: finalBuffer)
        
        let outputStartAddress = finalBuffer.baseAddress!.advanced(by: inputBuffer.count)
        let outputWriteBuffer = UnsafeMutableRawBufferPointer(start: outputStartAddress, count: outputBuffer.count)
        
        outputBuffer.copyBytes(to: outputWriteBuffer)
        
        return UnsafeRawBufferPointer(finalBuffer)
    }
    
    /**
     * Decodes a training example from the base address of a buffer of data that encodes this training item, and updates the base address to point to the next
     * byte after this buffer
     *
     * - Parameter baseAddress: `UnsafeRawPointer` pointing to the beginning of a byte buffer that encodes a training item, which
     * will be incremented.
     */
    static func unsafeRead(from baseAddress: inout UnsafeRawPointer) -> DataSet.Item {
        DataSet.Item(fromUnsafeBaseAddress: &baseAddress)
    }
    
    /**
     * Encodes a training example to the base address of a buffer of data that will store this training item, and updates the base address to point to the next
     * byte after this buffer
     *
     * - Parameter baseAddress: `UnsafeRawPointer` pointing to the beginning of a byte buffer that will store a training item, which
     * will be incremented.
     */
    func unsafeWrite(to baseAddress: inout UnsafeMutableRawPointer) {
        input.unsafeWrite(to: &baseAddress)
        output.unsafeWrite(to: &baseAddress)
    }
    
}
