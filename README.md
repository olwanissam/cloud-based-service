# ☁️ Cloud-Based Distributed ML Service (PySpark)

This project is a cloud-native data processing service that enables users to perform distributed machine learning analytics via a web interface. It leverages Apache Spark for parallel processing and is deployed on a DigitalOcean AMD Droplet.

## 🚀 Features
* **Distributed ML Engine:** Executes four concurrent ML tasks (Regression, KMeans, FPGrowth, and Summary Aggregation).
* **Real-time Scalability Analysis:** Benchmarks performance across 1, 2, 4, and 8 worker configurations.
* **Automated Data Validation:** Infers schema, calculates descriptive statistics, and handles missing values.
* **Cloud Storage Integration:** Persistently stores dataset uploads and processing results.

## 🛠️ Tech Stack
* **Frontend:** Gradio
* **Backend:** PySpark (Apache Spark 3.5.1)
* **Infrastructure:** DigitalOcean 2-vCPU AMD Droplet
* **Process Management:** Linux Systemd

## 📊 Performance Summary
 Based on experimental evaluations on the cloud platform:

| Cluster Size | Avg Execution Time (sec) | Speedup | Efficiency |
| :--- | :--- | :--- | :--- |
| 1 Machine | 0.64 | 1.00 | 100% |
| 2 Machines | 0.63 | 1.02 | 51.0% |
| 4 Machines | 0.66 | 0.97 | 24.3% |
| 8 Machines | 0.65 | 0.98 | 12.3% |

 *Note: For the 1.4 MB test dataset, efficiency decreases beyond 2 workers due to thread management overhead exceeding computation time on virtualized hardware.*

## 📖 How to Use
1. **Access the Service:** Navigate to `http://209.38.219.47:7860`.
2. **Upload Data:** Upload a valid `.csv` dataset.
3. **Execute:** Click "Start Distributed Processing".
4. **Analyze:** View the generated descriptive statistics and scalability graph directly on the dashboard.

## 📝 Configuration
The application is configured to run as a system service for 24/7 availability.
-  **Spark Home:** `/root/spark-3.5.1-bin-hadoop3` 
-  **Storage Path:** `/root/spark_project/cloud_storage`
