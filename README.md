# Spark MNIST

The Spark implementation of an ANN running the MNIST dataset.

## ANN

The used ANN is `bgreeven`'s ANN implementation in Spark. It has yet to be merged
into Spark-MLlib, but it's available as code for now.

## Usage

Acquire the MNIST dataset in a TSV format. Then, run the following command:

    cd spark-mnist
    sbt package
    cd target/scala-2.10
    spark-submit --name "Spark MNIST NN" --master [local|yarn-cluster] --class Mnist mnist_1.0-2.10.jar {train} {test} {output}

For the `master` option, pass the corresponding master, either `local` to run it local, or `yarn-cluster` to run it on a Hadoop cluster.

