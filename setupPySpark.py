# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 18:20:47 2017

@author: naveen.nathan
"""

import sys
from os import environ
from os.path import join

environ['SPARK_HOME'] = "C:\spark-2.3.0-bin-hadoop2.7"
SPARK_HOME = environ['SPARK_HOME']
sys.path.append(join(SPARK_HOME, "python"))
sys.path.append(join(SPARK_HOME, "python", "lib"))
sys.path.append(join(SPARK_HOME,"python", "lib", "pyspark.zip"))
sys.path.append(join(SPARK_HOME,"python", "lib", "py4j-0.10.4-src.zip"))

from pyspark import SparkContext, SparkConf

conf=SparkConf()
conf.set("spark.executor.memory", "8g")
conf.set("spark.cores.max", "4")
sc = SparkContext('local', conf = conf)
