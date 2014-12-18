package org.apache.spark.mllib.ann

/**
 * Created by tolga on 4-12-14.
 */
trait ANNHelper {
  protected val topology: Array[Int]
  protected def g(x: Double) = 1.0 / (1.0 + math.exp(-x))
  protected val L = topology.length - 1
  protected val noWeights = {
    var tmp = 0
    var l = 1
    while (l <= L) {
      tmp = tmp + topology(l) * (topology(l - 1) + 1)
      l += 1
    }
    tmp
  }
  protected val ofsWeight: Array[Int] = {
    val tmp = new Array[Int](L + 1)
    var curPos = 0
    tmp(0) = 0
    var l = 1
    while (l <= L) {
      tmp(l) = curPos
      curPos = curPos + (topology(l - 1) + 1) * topology(l)
      l += 1
    }
    tmp
  }
  protected val noNodes: Int = {
    var tmp: Integer = 0
    var l = 0
    while (l < topology.size) {
      tmp = tmp + topology(l)
      l += 1
    }
    tmp
  }
  protected val ofsNode: Array[Int] = {
    val tmp = new Array[Int](L + 1)
    tmp(0) = 0
    var l = 1
    while (l <= L) {
      tmp(l) = tmp(l - 1) + topology(l - 1)
      l += 1
    }
    tmp
  }

  protected def forwardRun(arrData: Array[Double], arrWeights: Array[Double]): Array[Double] = {
    val arrNodes = new Array[Double](noNodes)
    var i: Int = 0
    var j: Int = 0
    var l: Int = 0
    i = 0
    while (i < topology(0)) {
      arrNodes(i) = arrData(i)
      i += 1
    }
    l = 1
    while (l <= L) {
      j = 0
      while (j < topology(l)) {
        var cum: Double = 0.0
        i = 0
        while (i < topology(l - 1)) {
          cum = cum +
            arrWeights(ofsWeight(l) + (topology(l - 1) + 1) * j + i) *
              arrNodes(ofsNode(l - 1) + i)
          i += 1
        }
        cum = cum + arrWeights(ofsWeight(l) + (topology(l - 1) + 1) * j + topology(l - 1))
        arrNodes(ofsNode(l) + j) = g(cum)
        j += 1
      }
      l += 1
    }
    arrNodes
  }
}
