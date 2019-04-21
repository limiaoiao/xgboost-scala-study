import ml.dmlc.xgboost4j.scala.spark.{XGBoost, XGBoostRegressor}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{LabeledPoint, VectorAssembler}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.rdd.{PairRDDFunctions, RDD}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}

object predictSalary {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder.master("local").appName("example").getOrCreate()

    val sc = spark.sparkContext

    val trainFeaturePath = "DataScientist_Homework/train_features.csv"
    val trainLabelPaht = "DataScientist_Homework/train_salaries.csv"

    val testingFeaturePath = "DataScientist_Homework/test_features.csv"

    val (codedTrainDataset, testFeatureDataset) = loadTrainDataset(trainFeaturePath, trainLabelPaht, testingFeaturePath, sc)

    val featureWithLabelSchema = new StructType(Array(
      StructField("company id", IntegerType, true),
      StructField("job type", IntegerType, true),
      StructField("degree", IntegerType, true),
      StructField("major", IntegerType, true),
      StructField("industry", IntegerType, true),
      StructField("years experience", IntegerType, true),
      StructField("miles from metropolis", IntegerType, true),
      StructField("salary", DoubleType, true)))

    val featureSchema = new StructType(Array(
      StructField("company id", IntegerType, true),
      StructField("job type", IntegerType, true),
      StructField("degree", IntegerType, true),
      StructField("major", IntegerType, true),
      StructField("industry", IntegerType, true),
      StructField("years experience", IntegerType, true),
      StructField("miles from metropolis", IntegerType, true)))

    val codedTrainDF = spark.createDataFrame(codedTrainDataset, featureWithLabelSchema)

    val testFeatureDF = spark.createDataFrame(testFeatureDataset, featureSchema)

    codedTrainDF.show()
    testFeatureDF.show()

    val vectorAssembler = new VectorAssembler().
      setInputCols(Array("company id", "job type", "degree", "major", "industry", "years experience", "miles from metropolis")).
      setOutputCol("features")
    val xgbInput = vectorAssembler.transform(codedTrainDF).select("features",
      "salary")

    val Array(train, eval1, eval2, test) = xgbInput.randomSplit(Array(0.6, 0.2, 0.1, 0.1))
    train.persist()
    eval1.persist()
    eval2.persist()
    test.persist()

    val param =
      Map("eta" -> 0.4f,
        "max_depth" -> 5,
        "num_round" -> 100,
//        "num_workers" -> 2,
        "eval_metric" -> "rmse",
        "eval_sets" -> Map("eval1" -> eval1, "eval2" -> eval2)
      )

    //
    //    val paramGrid = new ParamGridBuilder()
    //      .addGrid(booster.maxDepth, Array(3, 8))
    //      .addGrid(booster.eta, Array(0.2, 0.6))
    //      .build()
    //    val cv = new CrossValidator()
    //      .setEstimator(pipeline)
    //      .setEvaluator(evaluator)
    //      .setEstimatorParamMaps(paramGrid)
    //      .setNumFolds(3)

    val xgbReg = new XGBoostRegressor(param).
      setFeaturesCol("features").
      setLabelCol("salary")

    val xgbModel = xgbReg.fit(train)
    val results = xgbModel.transform(test)
    results.show()

    val xgbClassificationModelPath = "/tmp/xgbClassificationModel"
    xgbModel.save(xgbClassificationModelPath)

    train.unpersist()
    eval1.unpersist()
    eval2.unpersist()
    test.unpersist()

  }

  //  RDD[LabeledPoint]
  def loadTrainDataset(featurePath: String, labelPath: String, testPath: String, sc: SparkContext):
  (RDD[Row], RDD[Row]) = {

    val inputFeatureDataset = sc.textFile(featurePath).zipWithIndex().map(i => (i._2, i._1))
    val inputLabelDataset = sc.textFile(labelPath).zipWithIndex().map(i => (i._2, i._1))
    val zippedDataset = inputFeatureDataset.join(inputLabelDataset).map(i => {
      val label = i._2._2.split(",")(1)
      i._2._1.split(",") ++ Array(label)

    })

    val zippedWithoutHeader = zippedDataset.filter(i => i(0) != "jobId")


    val inputTestFeatureDataset = sc.textFile(testPath).map(i => i.split(","))
    val testWithoutHeader = inputTestFeatureDataset.filter(i => i(0) != "jobId")


    val companyID = zippedWithoutHeader.map(i => i(1)).collect().distinct.drop(0)
    val major = zippedWithoutHeader.map(i => i(4)).collect().distinct.drop(0)
    val industry = zippedWithoutHeader.map(i => i(5)).collect().distinct.drop(0)

    //naturally speaking, these two rank array could be ordered by calculating mean salary
    //here I just order it in  general consideration order
    val jobTypeRank = Array("CEO", "VICE_PRESIDENT", "CTO", "CFO", "MANAGER", "SENIOR", "JUNIOR", "JANITOR").reverse
    val degreeRank = Array("NONE", "HIGH_SCHOOL", "BACHELORS", "MASTERS", "DOCTORAL")

    val trainWithLabel = zippedWithoutHeader.map(i => {
      Row(companyID.indexOf(i(1)),
        jobTypeRank.indexOf(i(2)),
        degreeRank.indexOf(i(3)),
        major.indexOf(i(4)),
        industry.indexOf(i(5)),
        i(6).toInt,
        i(7).toInt,
        i(8).toDouble)
    })

    val testFeature = testWithoutHeader.map(i => {
      Row(companyID.indexOf(i(1)),
        jobTypeRank.indexOf(i(2)),
        degreeRank.indexOf(i(3)),
        major.indexOf(i(4)),
        industry.indexOf(i(5)),
        i(6).toInt,
        i(7).toInt)
    })

    (trainWithLabel, testFeature)

  }
}
