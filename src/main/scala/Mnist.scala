import org.apache.spark.mllib.ann.ArtificialNeuralNetwork
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

object Mnist extends App {
  private val SEPERATOR = "\t"
  private val SPLITS    = 24

  val conf = new SparkConf().setAppName("MNIST Neural Network")
  val sc = new SparkContext(conf)

  val trainImages = sc.textFile(args(0), SPLITS)
      .map(_ split SEPERATOR).cache
  val testImages  = sc.textFile(args(1), SPLITS)
      .map(_ split SEPERATOR).cache

  val (trainValues, trainLabels) = splitValuesLabels(trainImages)
  val train = trainValues.zip(trainLabels).cache

  val (testValues, testLabels) = splitValuesLabels(testImages)

  val startTime = System.currentTimeMillis
  val network   = ArtificialNeuralNetwork.train(train, Array[Int](300), 500)
  val endTime   = System.currentTimeMillis

  val prediction = network.predict(testValues).map(_._2).cache
  val output     = prediction.zip(testLabels).cache
  val errRate    = output.map(T =>
      (T._2.toArray(0) - T._1.toArray(0)) * (T._2.toArray(0) - T._1.toArray(0))
    ).reduce((u, v) => u + v)
//  output.saveAsTextFile(args(2))

  println(s"Elapsed training time: ${(endTime - startTime) / 1000}s. Error rate: $errRate.")

  def indexOfMax(values: Array[Double]) = {
    val max = values.max
    values indexOf max
  }

  def splitValuesLabels(valLab: RDD[Array[String]]): (RDD[Vector], RDD[Vector]) = {
    val values = valLab.map(e => Vectors.dense(e.init.map(_.toDouble)))
    val labels = valLab.map {
      e =>
        val labs = Array.ofDim[Double](10)
        labs(e.last.toInt) = 1D
        Vectors.dense(labs)
    }
    (values.cache, labels.cache)
  }
}
