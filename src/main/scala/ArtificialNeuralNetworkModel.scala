/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.ann

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

/*
 * Implements a Artificial Neural Network (ANN)
 *
 * The data consists of an input vector and an output vector, combined into a single vector
 * as follows:
 *
 * [ ---input--- ---output--- ]
 *
 * NOTE: output values should be in the range [0,1]
 *
 * For a network of H hidden layers:
 *
 * hiddenLayersTopology(h) indicates the number of nodes in hidden layer h, excluding the bias
 * node. h counts from 0 (first hidden layer, taking inputs from input layer) to H - 1 (last
 * hidden layer, sending outputs to the output layer).
 *
 * hiddenLayersTopology is converted internally to topology, which adds the number of nodes
 * in the input and output layers.
 *
 * noInput = topology(0), the number of input nodes
 * noOutput = topology(L-1), the number of output nodes
 *
 * input = data( 0 to noInput-1 )
 * output = data( noInput to noInput + noOutput - 1 )
 *
 * W_ijl is the weight from node i in layer l-1 to node j in layer l
 * W_ijl goes to position ofsWeight(l) + j*(topology(l-1)+1) + i in the weights vector
 *
 * B_jl is the bias input of node j in layer l
 * B_jl goes to position ofsWeight(l) + j*(topology(l-1)+1) + topology(l-1) in the weights vector
 *
 * error function: E( O, Y ) = sum( O_j - Y_j )
 * (with O = (O_0, ..., O_(noOutput-1)) the output of the ANN,
 * and (Y_0, ..., Y_(noOutput-1)) the input)
 *
 * node_jl is node j in layer l
 * node_jl goes to position ofsNode(l) + j
 *
 * The weights gradient is defined as dE/dW_ijl and dE/dB_jl
 * It has same mapping as W_ijl and B_jl
 *
 * For back propagation:
 * delta_jl = dE/dS_jl, where S_jl the output of node_jl, but before applying the sigmoid
 * delta_jl has the same mapping as node_jl
 *
 * Where E = ((estOutput-output),(estOutput-output)),
 * the inner product of the difference between estimation and target output with itself.
 *
 */

/**
 * Artificial neural network (ANN) model
 *
 * @param weights the weights between the neurons in the ANN.
 * @param topology array containing the number of nodes per layer in the network, including
 * the nodes in the input and output layer, but excluding the bias nodes.
 */
class ArtificialNeuralNetworkModel (val weights: Vector, val topology: Array[Int])
  extends Serializable with ANNHelper {

  /**
   * Predicts values for a single data point using the trained model.
   *
   * @param testData represents a single data point.
   * @return prediction using the trained model.
   */
  def predict(testData: Vector): Vector = {
    Vectors.dense(computeValues(testData.toArray, weights.toArray))
  }

  /**
   * Predict values for an RDD of data points using the trained model.
   *
   * @param testDataRDD RDD representing the input vectors.
   * @return RDD with predictions using the trained model as (input, output) pairs.
   */
  def predict(testDataRDD: RDD[Vector]): RDD[(Vector,Vector)] = {
    testDataRDD.map(T => (T, predict(T)) )
  }

  private def computeValues(arrData: Array[Double], arrWeights: Array[Double]): Array[Double] = {
    val arrNodes = forwardRun(arrData, arrWeights)
    arrNodes.slice(arrNodes.size - topology(L), arrNodes.size)
  }
}
