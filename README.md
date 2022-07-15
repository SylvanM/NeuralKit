# NeuralKit

## Summary

NeuralKit is a pure Swift implementation of deep learning algorithms for whatever one may find this useful for,
using [MatrixKit](https://github.com/SylvanM/MatrixKit) to perform the linear algebra associated with deep learning.

This library is **not** complete at all, it is in its infancy! I encourage anyone interested to use this library
as they see fit, and if they see any issues (of which I'm sure there are a ton) please let me know!

## Why?

[Marcin Krzyzanowski](https://github.com/krzyzanowskim) said it best: "[Because I can.](https://github.com/krzyzanowskim/CryptoSwift/issues/5#issuecomment-53379391)"

For a while I've wanted to just make a neural network library for fun and to learn how to do it, so when some friends of mine
wanted to work on our own project that would require some serious machine learning, I figured now would be the perfect time to really develop
and perfect this library for real big practical usage. But, a lot of Apple's ML API's available in Swift are quite hard to use,
and using other libraries just aren't that practical when you don't have a nice graphics card or a supercomputer (of which I have neither).
Not to mention, there are a lot of things you just can't do with Apple's current available ML API's!

I was inspired by [CryptoSwift](https://github.com/krzyzanowskim/CryptoSwift), as it provided a much more
user friendly way of running crypto routines in pure Swift that is in some ways more versatile than `CommonCrypto`.

So, the vision for this project is that regular people without graphics cards, supercomputers, or a lot of ML experience,
can create ML models for their own projects, with those computations being easily able to be run in parralel with any 
other computer that can compile this package. For example, want to do some deep learning and have some friends also interested,
but don't have beefy computers and ML experience? The idea of this package is that you'd be able to easily create an ML model and
combine the computing power of many modest computers to have practical data science usage.

## Installation

This is installed like any other Swift package, though it has the additional dependency [MatrixKit](https://github.com/SylvanM/MatrixKit),
which should automatically be taken care of by the Swift package manager.

## Quick Start Example

A common "Hello World" of machine learning is getting a neural network to recognize handwritten digits, often using the 
[MNIST](http://yann.lecun.com/exdb/mnist/) database. There is sample code in `Tests/NeuralKitTests/Examples/GDTestsDigits.swift`
that uses gradient descent to train a neural network to perform this task. Since the current implementation of NeuralKit uses
a bespoke file type for data sets, I've included files for converting the MNIST database to NeuralKit Data Set (`.nkds`) files,
and the NKDS version of the MNIST database is included in the most recent release of this package. Feel free to use that!

