import org.apache.spark.mllib.ann.ArtificialNeuralNetwork
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

object Mnist extends App {
  private final val SEPERATOR = "\t"
  private final val HADOOP_SPLITS = 24

//  Create Spark configuration and use it to create a SparkContext
  val conf = new SparkConf().setAppName("MNIST Neural Network")
  val sc = new SparkContext(conf)

//  Using the SparkContext, read the training (arg(0)) set and test (arg(1)) set.
//  After reading it, immediately map over each element and split it on a tab seperator.
//  This data set is a tab seperated file.
  val trainImages = sc.textFile(args(0), HADOOP_SPLITS)
      .map(_ split SEPERATOR).cache
  val testImages  = sc.textFile(args(1), HADOOP_SPLITS)
      .map(_ split SEPERATOR).cache

//  Split the the values and the labels from the training set.
//  "splitValuesLabels" also converts all values to numeric values.
  val (trainValues, trainLabels) = splitValuesLabels(trainImages)
//  Zip the training values and training labels together.
  val train = trainValues.zip(trainLabels).cache

//  Split the values and the labels from the test set.
  val (testValues, testLabels) = splitValuesLabels(testImages)

  val startTime = System.currentTimeMillis
//  Train the ANN with the training set "train",
//  with a single hidden layer of 100 neurons,
//  and a maximum iteration of 50.
  val network   = ArtificialNeuralNetwork.train(train, Array[Int](100), 50)
  val endTime   = System.currentTimeMillis

//  Predict the test set using the trained ANN.
//  Immediately map the ouput and only keep the output labels "_._2"
  val prediction = network.predict(testValues).map(_._2).cache
  val output     = prediction.zip(testLabels).cache
  val n          = output.count
  val errRate    = output.map {
    T =>
      val p = T._2.toArray
      val l = T._1.toArray
//      Vectorized solution of 10 dimensional squared error
      (p(0) - l(0)) * (p(0) - l(0)) +
        (p(1) - l(1)) * (p(1) - l(1)) +
        (p(2) - l(2)) * (p(2) - l(2)) +
        (p(3) - l(3)) * (p(3) - l(3)) +
        (p(4) - l(4)) * (p(4) - l(4)) +
        (p(5) - l(5)) * (p(5) - l(5)) +
        (p(6) - l(6)) * (p(6) - l(6)) +
        (p(7) - l(7)) * (p(7) - l(7)) +
        (p(8) - l(8)) * (p(8) - l(8)) +
        (p(9) - l(9)) * (p(9) - l(9))
  }.reduce((u, v) => u + v)
//  Optionally save the prediction output to the preferred location
//  output.saveAsTextFile(args(2))

  println(s"Elapsed training time: ${(endTime - startTime) / 1000}s. Error rate: ${errRate / n}.")

//  This function converts the label output of the ANN to a single value.
//  The output is a 10 dimensional vector of probabilities for each digit
//  between 0-9. The highest probability is chosen.
  def indexOfMax(values: Array[Double]) = {
    val max = values.max
    values indexOf max
  }

  def splitValuesLabels(valLab: RDD[Array[String]]): (RDD[Vector], RDD[Vector]) = {
    val values = valLab.map(e => Vectors.dense(e.init.map(_.toDouble / 255D)))
    val labels = valLab.map {
      e =>
        val labs = Array.ofDim[Double](10)
        labs(e.last.toInt) = 1D
        Vectors.dense(labs)
    }
    (values.cache, labels.cache)
  }
}
