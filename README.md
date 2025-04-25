The rapid growth of the Indian semiconductor industry has necessitated the 
optimization of microfabrication processes to enhance efficiency, yield, and quality in 
Integrated Circuit (IC) manufacturing. Microfabrication involves intricate steps including 
photolithography, etching, doping, and deposition, where process variability can significantly 
impact wafer quality and production cost. This project focuses on data-driven optimization of 
semiconductor wafer production by leveraging machine learning (ML) techniques to predict 
and minimize defects, process deviations, and resource wastage. The study begins with 
comprehensive Exploratory Data Analysis (EDA) to understand the distribution, correlation, 
and patterns among critical microfabrication parameters such as temperature, pressure, etch 
time, gas flow rates, and resistivity measurements. Visualization techniques including 
heatmaps, box plots, and distribution curves were employed to uncover hidden trends and 
outliers that may affect model performance and production outcomes. The existing system 
utilizes an Extra Trees Classifier to model and classify wafer production outcomes based on 
historical process data. While this model offers robustness and speed, it demonstrates 
limitations in generalization and interpretability when applied to complex manufacturing 
datasets. To address these limitations, a Random Forest Classifier is proposed as an improved 
solution. This ensemble learning method enhances predictive performance by aggregating the 
results of multiple decision trees and reducing variance. Comparative evaluation between the 
Extra Trees and Random Forest classifiers was conducted using key performance metrics such 
as accuracy, precision, recall, F1-score, and confusion matrix analysis. Experimental results 
indicate that the Random Forest Classifier outperforms the Extra Trees Classifier in terms of 
prediction accuracy and robustness across varied process conditions. The proposed approach 
not only improves yield forecasting but also supports proactive decision-making in real-time 
microfabrication process control. The optimized machine learning framework provides a 
scalable solution for Indian IC manufacturers, enabling smarter process management, reduced 
operational costs, and advancement towards Industry 4.0 compliance in semiconductor 
production.
