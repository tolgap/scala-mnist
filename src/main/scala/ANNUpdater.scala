package org.apache.spark.mllib.ann

import breeze.linalg.{Vector => BV, axpy => brzAxpy}

import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.optimization.Updater

/**
 * Created by tolga on 4-12-14.
 */
class ANNUpdater extends Updater {

  override def compute(
                        weightsOld: Vector,
                        gradient: Vector,
                        stepSize: Double,
                        iter: Int,
                        regParam: Double): (Vector, Double) = {
    val thisIterStepSize = stepSize
    val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    brzAxpy(-thisIterStepSize, gradient.toBreeze, brzWeights)
    (Vectors.fromBreeze(brzWeights), 0)
  }
}
