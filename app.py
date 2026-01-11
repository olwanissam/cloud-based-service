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

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/root/spark-3.5.1-bin-hadoop3"
findspark.init()

CLOUD_STORAGE = "/root/spark_project/cloud_storage"
os.makedirs(CLOUD_STORAGE, exist_ok=True)

def process_and_benchmark(file_obj):
    if file_obj is None: return None, None, "No file uploaded."
    
    file_path = file_obj.name
    results_output = []
    
    spark_stat = SparkSession.builder.master("local[2]").getOrCreate()
    df = spark_stat.read.csv(file_path, header=True, inferSchema=True).dropna()
    
    rows = df.count()
    cols = len(df.columns)
    null_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0].asDict()
    avg_null = sum(null_counts.values()) / (rows * cols) * 100
    
    stats_summary = f"""
    ### 📊 Dataset Descriptive Statistics
    - **Total Rows:** {rows}
    - **Total Columns:** {cols}
    - **Data Types:** {dict(df.dtypes)}
    - **Missing Value Percentage:** {avg_null:.2f}%
    """
    spark_stat.stop()

    worker_counts = [1, 2, 4, 8]
    exp_results = []
    
    for n in worker_counts:
        spark = SparkSession.builder.master(f"local[{n}]").getOrCreate()
        
        # Prepare Data
        numeric_cols = [f for (f, dtype) in df.dtypes if dtype in ("int", "double", "float")]
        assembler = VectorAssembler(inputCols=numeric_cols[:-1], outputCol="features", handleInvalid="skip")
        ml_data = assembler.transform(df).select("features", numeric_cols[-1]).cache()
        
        start_time = time.time()
        
        # --- JOB 1: Regression (Decision Tree) ---
        dt = DecisionTreeRegressor(featuresCol="features", labelCol=numeric_cols[-1])
        model_reg = dt.fit(ml_data)
        
        # --- JOB 2: Clustering (KMeans) ---
        kmeans = KMeans().setK(3).setSeed(1)
        model_km = kmeans.fit(ml_data)
        
        # --- JOB 3: Time-Series/Aggregation Summary ---
        summary = df.describe().toPandas()
        
        # --- JOB 4: Frequent Itemsets (FPGrowth) ---
        # Note: We simulate this by grouping a numeric col into 'items'
        fp_data = df.select(F.array(numeric_cols[0]).alias("items"))
        fp = FPGrowth(itemsCol="items", minSupport=0.2, minConfidence=0.5)
        model_fp = fp.fit(fp_data)
        
        duration = time.time() - start_time
        exp_results.append({"Workers (N)": n, "Time (s)": round(duration, 2)})
        
        summary.to_csv(f"{CLOUD_STORAGE}/summary_n{n}.csv")
        
        spark.stop()

    report_df = pd.DataFrame(exp_results)
    t1 = report_df.iloc[0]['Time (s)']
    report_df['Speedup'] = t1 / report_df['Time (s)']
    report_df['Efficiency (%)'] = (report_df['Speedup'] / report_df['Workers (N)']) * 100
    
    plt.figure(figsize=(8, 5))
    plt.plot(report_df['Workers (N)'], report_df['Speedup'], marker='o', label="Measured Speedup")
    plt.plot(report_df['Workers (N)'], report_df['Workers (N)'], '--', label="Ideal")
    plt.legend(); plt.grid(True)
    plot_path = "/root/spark_project/scalability_plot.png"
    plt.savefig(plot_path)
    
    return stats_summary, report_df, plot_path

with gr.Blocks() as demo:
    gr.Markdown("# ☁️ Cloud Distributed ML Service (PySpark)")
    file_input = gr.File(label="Upload Dataset (CSV)")
    run_btn = gr.Button("Start Distributed Processing", variant="primary")
    
    stats_out = gr.Markdown()
    with gr.Row():
        table_out = gr.DataFrame(label="Execution Metrics")
        plot_out = gr.Image(label="Scalability Graph")

    run_btn.click(process_and_benchmark, inputs=file_input, outputs=[stats_out, table_out, plot_out])

demo.launch(server_name="0.0.0.0", server_port=7860)
