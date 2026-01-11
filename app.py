import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import findspark
import gradio as gr
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.clustering import KMeans
from pyspark.ml.fpm import FPGrowth


os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64" # should attach relative path of java jdk
os.environ["SPARK_HOME"] = "/spark/spark-3.5.1-bin-hadoop3" # should attach relative path of spark engine
findspark.init()

CLOUD_STORAGE = "./cloud_storage"
os.makedirs(CLOUD_STORAGE, exist_ok=True)

def process_and_benchmark(file_obj):
    if file_obj is None: return None, None, None, "No file uploaded."
    
    file_path = file_obj.name
    
    spark = SparkSession.builder.master("local[2]").appName("StatsEngine").getOrCreate()
    try:
        df = spark.read.csv(file_path, header=True, inferSchema=True).dropna()
        
        rows = df.count()
        cols = len(df.columns)
        null_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0].asDict()
        avg_null = (sum(null_counts.values()) / (rows * cols)) * 100
        
        unique_sample = str({c: df.select(c).distinct().count() for c in df.columns[:2]})

        stats_df = pd.DataFrame({
            "Metric": ["Total Rows", "Total Columns", "Null Percentage", "Unique Value Sample"],
            "Value": [rows, cols, f"{avg_null:.2f}%", unique_sample]
        })
        stats_df.to_csv(f"{CLOUD_STORAGE}/descriptive_stats.csv", index=False)
    finally:
        spark.stop()

    worker_counts = [1, 2, 4, 8]
    exp_results = []
    fp_display_df = pd.DataFrame()
    
    numeric_cols = [f for (f, dtype) in df.dtypes if dtype in ("int", "double", "float")]

    for n in worker_counts:
        spark_job = SparkSession.builder.master(f"local[{n}]").appName(f"ML_Job_{n}").getOrCreate()
        try:
            df_job = spark_job.read.csv(file_path, header=True, inferSchema=True).dropna()
            assembler = VectorAssembler(inputCols=numeric_cols[:2], outputCol="features")
            ml_data = assembler.transform(df_job).select("features", numeric_cols[0]).cache()
            
            start = time.time()
            
            # Job 1: Regression
            DecisionTreeRegressor(featuresCol="features", labelCol=numeric_cols[0]).fit(ml_data)
            # Job 2: KMeans
            KMeans(k=3).fit(ml_data)
            # Job 3: Summary Aggregation
            summary = df_job.describe().toPandas()
            summary.to_csv(f"{CLOUD_STORAGE}/summary_n{n}.csv")
            # Job 4: FPGrowth (Frequent Itemsets)
            fp_data = df_job.select(F.array(numeric_cols[0]).alias("items"))
            model_fp = FPGrowth(itemsCol="items", minSupport=0.2).fit(fp_data)
            
            if n == 1: # Capture first run for screen display
                fp_display_df = model_fp.freqItemsets.limit(10).toPandas()
                fp_display_df.to_csv(f"{CLOUD_STORAGE}/fp_results.csv", index=False)

            duration = time.time() - start
            exp_results.append({"Workers": n, "Time (s)": round(duration, 3)})
        finally:
            spark_job.stop()

    perf_df = pd.DataFrame(exp_results)
    perf_df['Speedup'] = perf_df.iloc[0]['Time (s)'] / perf_df['Time (s)']
    perf_df['Efficiency (%)'] = (perf_df['Speedup'] / perf_df['Workers']) * 100

    plt.figure(figsize=(8, 5))
    plt.plot(perf_df['Workers'], perf_df['Speedup'], marker='o', label="Actual")
    plt.plot(perf_df['Workers'], perf_df['Workers'], '--', color='red', label="Ideal")
    plt.xlabel("Cluster Size (N)"); plt.ylabel("Speedup (T1/Tn)"); plt.grid(True); plt.legend()
    plot_path = "/root/spark_project/scalability_plot.png"
    plt.savefig(plot_path)
    plt.close()
    
    return stats_df, fp_display_df, perf_df, plot_path

# --- FRONTEND layout ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ☁️ Cloud Distributed Data Processing")
    file_input = gr.File(label="Upload Dataset")
    run_btn = gr.Button("🚀 Execute Jobs & Benchmark", variant="primary")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 Descriptive Stats (Stored in Cloud)")
            stats_out = gr.DataFrame()
        with gr.Column():
            gr.Markdown("### 🌿 FPGrowth Frequent Itemsets")
            fp_out = gr.DataFrame()

    with gr.Row():
        perf_out = gr.DataFrame(label="Scalability Table")
        plot_out = gr.Image(label="Speedup Analysis")

    run_btn.click(process_and_benchmark, inputs=file_input, outputs=[stats_out, fp_out, perf_out, plot_out])

demo.launch(server_name="0.0.0.0", server_port=7860)
