package org.apache.spark.mllib.ann

import breeze.linalg.DenseVector
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.optimization.LBFGS
import org.apache.spark.rdd.RDD
import org.apache.spark.util.random.XORShiftRandom

/**
 * Performs the training of an Artificial Neural Network (ANN)
 *
 * @param topology A vector containing the number of nodes per layer in the network, including
 * the nodes in the input and output layer, but excluding the bias nodes.
 * @param maxNumIterations The maximum number of iterations for the training phase.
 * @param convergenceTol Convergence tolerance for LBFGS. Smaller value for closer convergence.
 */
class ArtificialNeuralNetwork (
                                topology: Array[Int],
                                maxNumIterations: Int,
                                convergenceTol: Double)
  extends Serializable {

  private val gradient = new ANNLeastSquaresGradient(topology)
  private val updater = new ANNUpdater()
  private val optimizer = new LBFGS(gradient, updater).
    setConvergenceTol(convergenceTol).
    setNumIterations(maxNumIterations)

  /**
   * Trains the ANN model.
   * Uses default convergence tolerance 1e-4 for LBFGS.
   *
   * @param trainingRDD RDD containing (input, output) pairs for training.
   * @param initialWeights the initial weights of the ANN
   * @return ANN model.
   */
  def run(trainingRDD: RDD[(Vector, Vector)], initialWeights: Vector):
  ArtificialNeuralNetworkModel = {
    val data = trainingRDD.map(v =>
      (0.0,
        Vectors.fromBreeze(DenseVector.vertcat(
          v._1.toBreeze.toDenseVector,
          v._2.toBreeze.toDenseVector))
        ))
    val weights = optimizer.optimize(data, initialWeights)
    new ArtificialNeuralNetworkModel(weights, topology)
  }
}

/**
 * Top level methods for training the artificial neural network (ANN)
 */
object ArtificialNeuralNetwork {

  private val defaultTolerance: Double = 1e-4

  /**
   * Trains an ANN.
   * Uses default convergence tolerance 1e-4 for LBFGS.
   *
   * @param trainingRDD RDD containing (input, output) pairs for training.
   * @param hiddenLayersTopology number of nodes per hidden layer, excluding the bias nodes.
   * @param maxNumIterations specifies maximum number of training iterations.
   * @return ANN model.
   */
  def train(
             trainingRDD: RDD[(Vector, Vector)],
             hiddenLayersTopology: Array[Int],
             maxNumIterations: Int): ArtificialNeuralNetworkModel = {
    train(trainingRDD, hiddenLayersTopology, maxNumIterations, defaultTolerance)
  }

  /**
   * Continues training of an ANN.
   * Uses default convergence tolerance 1e-4 for LBFGS.
   *
   * @param trainingRDD RDD containing (input, output) pairs for training.
   * @param model model of an already partly trained ANN.
   * @param maxNumIterations maximum number of training iterations.
   * @return ANN model.
   */
  def train(
             trainingRDD: RDD[(Vector,Vector)],
             model: ArtificialNeuralNetworkModel,
             maxNumIterations: Int): ArtificialNeuralNetworkModel = {
    train(trainingRDD, model, maxNumIterations, defaultTolerance)
  }

  /**
   * Trains an ANN with given initial weights.
   * Uses default convergence tolerance 1e-4 for LBFGS.
   *
   * @param trainingRDD RDD containing (input, output) pairs for training.
   * @param initialWeights initial weights vector.
   * @param maxNumIterations maximum number of training iterations.
   * @return ANN model.
   */
  def train(
             trainingRDD: RDD[(Vector,Vector)],
             hiddenLayersTopology: Array[Int],
             initialWeights: Vector,
             maxNumIterations: Int): ArtificialNeuralNetworkModel = {
    train(trainingRDD, hiddenLayersTopology, initialWeights, maxNumIterations, defaultTolerance)
  }

  /**
   * Trains an ANN using customized convergence tolerance.
   *
   * @param trainingRDD RDD containing (input, output) pairs for training.
   * @param model model of an already partly trained ANN.
   * @param maxNumIterations maximum number of training iterations.
   * @param convergenceTol convergence tolerance for LBFGS. Smaller value for closer convergence.
   * @return ANN model.
   */
  def train(
             trainingRDD: RDD[(Vector,Vector)],
             model: ArtificialNeuralNetworkModel,
             maxNumIterations: Int,
             convergenceTol: Double): ArtificialNeuralNetworkModel = {
    new ArtificialNeuralNetwork(model.topology, maxNumIterations, convergenceTol).
      run(trainingRDD, model.weights)
  }

  /**
   * Continues training of an ANN using customized convergence tolerance.
   *
   * @param trainingRDD RDD containing (input, output) pairs for training.
   * @param hiddenLayersTopology number of nodes per hidden layer, excluding the bias nodes.
   * @param maxNumIterations maximum number of training iterations.
   * @param convergenceTol convergence tolerance for LBFGS. Smaller value for closer convergence.
   * @return ANN model.
   */
  def train(
             trainingRDD: RDD[(Vector, Vector)],
             hiddenLayersTopology: Array[Int],
             maxNumIterations: Int,
             convergenceTol: Double): ArtificialNeuralNetworkModel = {
    val topology = convertTopology(trainingRDD, hiddenLayersTopology)
    new ArtificialNeuralNetwork(topology, maxNumIterations, convergenceTol).
      run(trainingRDD, randomWeights(topology, false))
  }

  /**
   * Trains an ANN with given initial weights.
   *
   * @param trainingRDD RDD containing (input, output) pairs for training.
   * @param initialWeights initial weights vector.
   * @param maxNumIterations maximum number of training iterations.
   * @param convergenceTol convergence tolerance for LBFGS. Smaller value for closer convergence.
   * @return ANN model.
   */
  def train(
             trainingRDD: RDD[(Vector,Vector)],
             hiddenLayersTopology: Array[Int],
             initialWeights: Vector,
             maxNumIterations: Int,
             convergenceTol: Double): ArtificialNeuralNetworkModel = {
    val topology = convertTopology(trainingRDD, hiddenLayersTopology)
    new ArtificialNeuralNetwork(topology, maxNumIterations, convergenceTol).
      run(trainingRDD, initialWeights)
  }

  /**
   * Provides a random weights vector.
   *
   * @param trainingRDD RDD containing (input, output) pairs for training.
   * @param hiddenLayersTopology number of nodes per hidden layer, excluding the bias nodes.
   * @return random weights vector.
   */
  def randomWeights(
                     trainingRDD: RDD[(Vector,Vector)],
                     hiddenLayersTopology: Array[Int]): Vector = {
    val topology = convertTopology(trainingRDD, hiddenLayersTopology)
    return randomWeights(topology, false)
  }

  /**
   * Provides a random weights vector, using given random seed.
   *
   * @param trainingRDD RDD containing (input, output) pairs for later training.
   * @param hiddenLayersTopology number of nodes per hidden layer, excluding the bias nodes.
   * @param seed random generator seed.
   * @return random weights vector.
   */
  def randomWeights(
                     trainingRDD: RDD[(Vector,Vector)],
                     hiddenLayersTopology: Array[Int],
                     seed: Int): Vector = {
    val topology = convertTopology(trainingRDD, hiddenLayersTopology)
    return randomWeights(topology, true, seed)
  }

  /**
   * Provides a random weights vector, using given random seed.
   *
   * @param inputLayerSize size of input layer.
   * @param outputLayerSize size of output layer.
   * @param hiddenLayersTopology number of nodes per hidden layer, excluding the bias nodes.
   * @param seed random generator seed.
   * @return random weights vector.
   */
  def randomWeights(
                     inputLayerSize: Int,
                     outputLayerSize: Int,
                     hiddenLayersTopology: Array[Int],
                     seed: Int): Vector = {
    val topology = inputLayerSize +: hiddenLayersTopology :+ outputLayerSize
    return randomWeights(topology, true, seed)
  }

  private def convertTopology(
                               input: RDD[(Vector,Vector)],
                               hiddenLayersTopology: Array[Int] ): Array[Int] = {
    val firstElt = input.first
    firstElt._1.size +: hiddenLayersTopology :+ firstElt._2.size
  }

  def randomWeights(topology: Array[Int], useSeed: Boolean, seed: Int = 0): Vector = {
    val rand: XORShiftRandom =
      if( !useSeed ) new XORShiftRandom() else new XORShiftRandom(seed)
    var i: Int = 0
    var l: Int = 0
    val noWeights = {
      var tmp = 0
      var i = 1
      while (i < topology.size) {
        tmp = tmp + topology(i) * (topology(i - 1) + 1)
        i += 1
      }
      tmp
    }
    val initialWeightsArr = new Array[Double](noWeights)
    var pos = 0
    l = 1
    while (l < topology.length) {
      i = 0
      while (i < (topology(l) * (topology(l - 1) + 1))) {
        initialWeightsArr(pos) = (rand.nextDouble * 4.8 - 2.4) / (topology(l - 1) + 1)
        pos += 1
        i += 1
      }
      l += 1
    }
    Vectors.dense(initialWeightsArr)
  }
}
