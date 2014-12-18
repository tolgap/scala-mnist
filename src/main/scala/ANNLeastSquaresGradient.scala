package org.apache.spark.mllib.ann

import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.optimization.Gradient

/**
 * Created by tolga on 4-12-14.
 */
class ANNLeastSquaresGradient(val topology: Array[Int]) extends Gradient with ANNHelper {

  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val arrData = data.toArray
    val arrWeights = weights.toArray
    var i: Int = 0
    var j: Int = 0
    var l: Int = 0
    // forward run
    val arrNodes = forwardRun(arrData, arrWeights)
    val arrDiff = new Array[Double](topology(L))
    j = 0
    while (j < topology(L)) {
      arrDiff(j) = arrNodes(ofsNode(L) + j) - arrData(topology(0) + j)
      j += 1
    }
    var err: Double = 0
    j = 0
    while (j < topology(L)) {
      err = err + arrDiff(j) * arrDiff(j)
      j += 1
    }
    err = err * .5
    // back propagation
    val arrDelta = new Array[Double](noNodes)
    j = 0
    while (j < topology(L)) {
      arrDelta(ofsNode(L) + j) =
        arrDiff(j) *
          arrNodes(ofsNode(L) + j) * (1 - arrNodes(ofsNode(L) + j))
      j += 1
    }
    l = L - 1
    while (l > 0) {
      j = 0
      while (j < topology(l)) {
        var cum: Double = 0.0
        i = 0
        while (i < topology(l + 1)) {
          cum = cum +
            arrWeights(ofsWeight(l + 1) + (topology(l) + 1) * i + j) *
              arrDelta(ofsNode(l + 1) + i) *
              arrNodes(ofsNode(l) + j) * (1 - arrNodes(ofsNode(l) + j))
          i += 1
        }
        arrDelta(ofsNode(l) + j) = cum
        j += 1
      }
      l -= 1
    }
    // gradient
    val arrGrad = new Array[Double](noWeights)
    l = 1
    while (l <= L) {
      j = 0
      while (j < topology(l)) {
        i = 0
        while (i < topology(l - 1)) {
          arrGrad(ofsWeight(l) + (topology(l - 1) + 1) * j + i) =
            arrNodes(ofsNode(l - 1) + i) *
              arrDelta(ofsNode(l) + j)
          i += 1
        }
        arrGrad(ofsWeight(l) + (topology(l - 1) + 1) * j + topology(l - 1)) =
          arrDelta(ofsNode(l) + j)
        j += 1
      }
      l += 1
    }
    (Vectors.dense(arrGrad), err)
  }

  override def compute(
                        data: Vector,
                        label: Double,
                        weights: Vector,
                        cumGradient: Vector): Double = {
    val (grad, err) = compute(data, label, weights)
    cumGradient.toBreeze += grad.toBreeze
    err
  }
}
