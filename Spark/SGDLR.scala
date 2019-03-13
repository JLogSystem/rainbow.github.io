package scala.test

import breeze.linalg.{DenseVector, axpy}
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.{Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

class SGDLR {
  private var stepSize: Double = 1.0
  private var numIterations: Int = 100
  private var regParam: Double = 0.0
  private var miniBatchFraction: Double = 1.0
  private var convergenceTol: Double = 0.001

  def setStepSize(stepSize:Double):this.type ={
    this.stepSize=stepSize
    this
  }
  def setMiniBatchFraction(fraction: Double): this.type = {
    this.miniBatchFraction = fraction
    this
  }
  def setNumIterations(iters: Int): this.type = {
    this.numIterations = iters
    this
  }
  def setRegParam(regParam: Double): this.type = {
    this.regParam = regParam
    this
  }
  def setConvergenceTol(tolerance: Double): this.type = {
    this.convergenceTol = tolerance
    this
  }



  //传入数据集进行训练
  def  train(input:RDD[LabeledPoint]):LogisticRegressionModel = {
    //特征值的数量
    val numFeatures = input.map(_.features.size).first()
    //初始化权重+截距
    var weights = Vectors.dense(new Array[Double](numFeatures+1))
    //标准化数据=(原数据-均值)/标准差  数据归一化：均值和方差
    val scaler = new StandardScaler(withStd = true, withMean = false).fit(input.map(_.features))
    //将数据转换为key-value的形式并实现增加截距：x+1
    val data=input.map(item=>(item.label,MLUtils.appendBias(scaler.transform(item.features)))).cache()
    //获取到data的数量，判断是否为0
    var numDatas = data.count()
  //记录当前的权重向量和前一个权重向量 用于更新
    var previousWeights= Vectors.dense(new Array[Double](numFeatures+1))
    var currentWeights= Vectors.dense(new Array[Double](numFeatures+1))
    //开始迭代
    var converged = false
    var iterator = 1
    while (!converged && iterator<=numIterations){
      val bcWeights = data.context.broadcast(weights)
      val (gradientSum,miniBathSize) = data.sample(false,miniBatchFraction,42+iterator)
        .treeAggregate((DenseVector.zeros[Double](weights.size),0L))(
          seqOp = (c, v) => {
            // c: (grad,count), v: (label, features)
            //返回的是梯度下降 (h(x)-y)*x
            var v_2 = DenseVector[Double](v._2.copy.toArray)
            val margin =  -1.0 * DenseVector(bcWeights.value.toArray).dot(v_2)
            val multiplier = (1.0 / (1.0 + math.exp(margin))) - v._1
            v_2 :*= multiplier
            c._1 += v_2
            //c.1存储的是迭代的所有的误差值
            (c._1, c._2 + 1)
          },
          combOp = (c1, c2) => {
            // c: (grad, count)
            //分区之间求和，算得所有的分区
            (c1._1 += c2._1, c1._2 + c2._2)
          })
      //跟新权重
      if(miniBathSize>0){
        // w' = w - thisIterStepSize * (gradient + regParam * w)
        // w' = (1 - thisIterStepSize * regParam) * w - thisIterStepSize * gradient
        val thisIterStepSize = stepSize / math.sqrt(iterator)
        val brzWeights: DenseVector[Double] = new DenseVector(weights.copy.toArray)
        brzWeights :*= (1.0 - thisIterStepSize * regParam)
        axpy(-thisIterStepSize, gradientSum.toDenseVector, brzWeights)
        weights = Vectors.dense(brzWeights.data)

        previousWeights = currentWeights
        currentWeights = weights
        //判断迭代的权重误差值，来决定是否结束迭代
          var solutionVecDiff:Double = 0;
          for (i <- 0 until previousWeights.size){
            solutionVecDiff +=math.abs(previousWeights(i)- currentWeights(i))
          }
         // val solutionVecDiff: Double = norm(new DenseVector(previousWeights.toArray) - DenseVector(currentWeights.toArray))
          var currentVecDiff:Double  = 0
          for (i <- 0 until currentWeights.size){
            currentVecDiff +=math.abs(currentWeights(i))
          }
          converged = solutionVecDiff < convergenceTol * Math.max(currentVecDiff,1.0)
        }
      iterator +=1
    }
    //得到权重和截距
    val intercept = weights(weights.size-1)
    var weights_result = Vectors.dense((weights.toArray.slice(0, weights.size - 1)))
    //还原归一化权重
    weights_result = scaler.transform(weights_result)
    //创建模型
    new LogisticRegressionModel(weights_result,intercept)
  }

}
