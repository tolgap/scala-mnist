import org.apache.spark.mllib.ann.ArtificialNeuralNetwork
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkContext, SparkConf}

object Mnist extends App {
  private final val SEPERATOR = "\t"
  private final val SPLITS    = 28

  val conf = new SparkConf().setAppName("MNIST Neural Network")
  val sc = new SparkContext(conf)

  val trainImages = sc.textFile(args(0), SPLITS).cache
      .map(_ split SEPERATOR)
  val testImages  = sc.textFile(args(1), SPLITS).cache
      .map(_ split SEPERATOR)

  val trainValues = trainImages.map(e => Vectors.dense(e.tail.map(_.toDouble))).cache
  val trainLabels = trainImages.map {
    e => {
      val labs = Array.ofDim[Double](10)
      labs(e.last.toInt) = 1
      Vectors.dense(labs)
    }
  }
  val train = trainValues.zip(trainLabels.cache)
  val testValues = testImages.map(e => Vectors.dense(e.tail.map(_.toDouble)))
  val testLabels = testImages.map(e => Vectors.dense(e.last.toDouble))

  val network = ArtificialNeuralNetwork.train(train, Array[Int](300), 1000)

  val prediction = network.predict(testValues.cache).map(_._2.toArray)
  val pred = prediction.map(indexOfMax).cache
  val output = pred.zip(testLabels)
  output.saveAsTextFile(args(2))

  final def indexOfMax(values: Array[Double]) = {
    val max = values.max
    values.indexOf(max)
  }
}
