//
//  GeneticTrainer.swift
//  
//
//  Created by Sylvan Martin on 7/1/22.
//

import Foundation
import MatrixKit

/**
 * A generic genetic algorithm that evolves a species `S`.
 *
 * This class is **not** specific to training `NeuralNetwork`s but is designed so that it can be used for that.
 */
public class GenericGeneticAlgorithm<S> {
    
    /**
     * The type used to remember the score for a particular species without scoring a copy or reference to the species itself.
     */
    public typealias ScoreRecord = (index: Int, score: Double)

    /**
     * Performs genetic evolution on a population of `S`
     *
     * - Parameter organisms: The population of organisms to evolve.
     * - Parameter scores: The scores for each species, updated after each genetation.
     * - Parameter fitness: The percentage of the population which is replaced with offspring after each generation
     * - Parameter generations: The amount of generations to evolve through. If set to 0, no evolution will be performed.
     * - Parameter score: A closure that computes the score of a single `S`.
     * - Parameter breed: A closure that defines how the offspring of two `S` is produced
     *
     * - Precondition: `organisms.count >= 2`
     * - Precondition: `0 < fitness < 1`
     * - Precondition: `scores.count == organisms.count`
     */
    static public func evolve(organisms: inout [S], scores: inout [ScoreRecord], fitness: Double, generations: Int, score: (S) -> Double, breed: (S, S) -> S, uponGenerationCompletion genFinished: (() -> ())? = nil) {
        if generations == 0 { return }
        
        // keep track of which creature scores what, so that we can deal with them accordingly later (sounds menacing)
        
        for i in 0..<scores.count {
            scores[i].index = i
            scores[i].score = score(organisms[i])
        }
        
        scores.sort { s1, s2 in
            s1.score <= s2.score
        }
        
        let cutoffIndex = Int(fitness * Double(scores.count))
        
        // replace unfit individuals with offspring of fit parents
        for i in 0..<cutoffIndex {
            
            // find two fit parents
            let p1 = Int.random(in: cutoffIndex..<scores.count)
            var p2: Int
            repeat {
                p2 = Int.random(in: cutoffIndex..<scores.count)
            } while p1 == p2
            
            let father = organisms[scores[p1].index]
            let mother = organisms[scores[p2].index]
            
            organisms[scores[i].index] = breed(father, mother)
            
        }
        
        genFinished?()
        evolve(organisms: &organisms, scores: &scores, fitness: fitness, generations: generations - 1, score: score, breed: breed, uponGenerationCompletion: genFinished)
    }
}