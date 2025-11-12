# Loan Repayment Clustering App (KMeans + Manual F1 Score)

This project explores loan repayment behavior using **KMeans clustering** and provides a fully interactive **Streamlit dashboard** for EDA, model training, and evaluation.  
The goal is to analyze whether customers will **fully repay** their loan or **default (not fully paid)** using an unsupervised approach.

---

## Features

###  **Exploratory Data Analysis (EDA)**
The app provides several visualizations to understand customer behavior:
- Distribution of fully paid vs. not fully paid loans  
- Loan purpose distribution  
- FICO score distribution grouped by repayment status  
- Interest Rate vs. FICO scatterplot  
- Cluster visualization using color-coded scatterplot  

---

### 🤖 **KMeans Clustering**
- Uses **all numerical features** + dummy-encoded categorical features  
- Features are scaled using **StandardScaler**  
- User selects `k` (number of clusters) using an interactive slider  
- Model assigns a cluster label to each customer  
- Color-coded cluster scatterplots for better visualization  

---

### **Manual F1 Score Calculation**
A custom, user-defined F1 score function is implemented to:
- Manually compute precision, recall, and F1-score  
- Evaluate how well cluster labels align with actual repayment status  
- Provide deeper insight into model performance beyond sklearn functions  

---

### **F1 Score vs. k Visualization**
The app includes a graph showing how the F1 score changes as the number of clusters varies from **2 to 10**.  
This helps understand:
- Why `k = 2` is logical for a binary repayment outcome  
- How the model behaves under different cluster settings  

---



## 📂 **Project Structure**

