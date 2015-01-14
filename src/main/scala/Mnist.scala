import org.apache.spark.mllib.ann.ArtificialNeuralNetwork
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkContext, SparkConf}

object Mnist extends App {
  private final val SEPERATOR = "\t"
  private final val SPLITS    = 28

  val conf = new SparkConf().setAppName("MNIST Neural Network")
  val sc = new SparkContext(conf)

  val trainImages = sc.textFile(args(0), SPLITS)
      .map(_ split SEPERATOR).cache
  val testImages  = sc.textFile(args(1), SPLITS)
      .map(_ split SEPERATOR).cache

  val trainValues = trainImages.map(e => Vectors.dense(e.init.map(_.toDouble))).cache
  val trainLabels = trainImages.map {
    e => {
      val labs = Array.ofDim[Double](10)
      labs(e.last.toInt) = 1
      Vectors.dense(labs)
    }
  }.cache
  val train = trainValues.zip(trainLabels).cache
  val testValues = testImages.map(e => Vectors.dense(e.tail.map(_.toDouble))).cache
  val testLabels = testImages.map(e => Vectors.dense(e.last.toDouble)).cache

  val startTime = System.currentTimeMillis
  val network = ArtificialNeuralNetwork.train(train, Array[Int](300), 1000)
  val endTime = System.currentTimeMillis

  val prediction = network.predict(testValues).map(_._2.toArray).cache
  val pred       = prediction.map(indexOfMax).cache
  val output     = pred.zip(testLabels)
  val errRate    = output.map(T => (T._2.toArray(0) - T._1) * (T._2.toArray(0) - T._1)).reduce((u,v) => u + v)
//  output.saveAsTextFile(args(2))

  println("Elapsed training time: ${(endTime - startTime) / 1000}s. Error rate: ${errRate}.")

  final def indexOfMax(values: Array[Double]) = {
    val max = values.max
    values.indexOf(max)
  }
}
