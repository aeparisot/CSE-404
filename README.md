# Classifying URLs: Improved Model Results and Analysis on Phishing URL Classification Using Machine Learning

**Dev Jyoti Ghosh Arnab, Aniket Kumar, Aaron Nguyen, Isabel Omness, Prachurjo Das, Abigail Parisot**

*Michigan State University*
{arnabdev, kumara29, nguyenaa, omnessis, dasprach, parisot1}@msu.edu

---

## Abstract

We present updated results for a phishing URL classification system trained on the PhiUSIIL Phishing URL Dataset from the UC Irvine Machine Learning Repository. We implement and evaluate four baseline classifiers: Logistic Regression, Decision Tree, Random Forest, and a Neural Network using accuracy, precision, recall, and F1 score. Using a trimmed feature set to reduce leaky features, Logistic Regression yields about 0.86 test accuracy, the Decision Tree achieves about 0.945, Random Forest achieved about 0.928, and Neural Network achieved 0.981 accuracy. Feature importance and ablation analyses identify the most impactful and leaky features to be removed. We discuss data processing, model development, and opportunities for future work.

---

## 1. Introduction

Phishing attacks remain one of the most prevalent forms of cybercrime, targeting users through deceptive URLs designed to mimic legitimate websites. These attacks can lead to identity theft, financial loss, and malware infection, disproportionately affecting users with limited technical literacy. The scale and adaptability of modern phishing campaigns motivate the development of automated, machine-learning based detection systems.

In this project, we train and evaluate multiple classifiers on the PhiUSIIL dataset, a large, feature-rich collection including features derived from both the URL string and the HTML source code of corresponding webpages. We first studied simple models and then moved to a neural network model. This paper reports on data collection, annotation, and processing, model and pipeline development, performance analysis and conclusions, as well as opportunities for future work.

---

## 2. Related Work

Phishing detection has evolved from static blacklist-based systems toward machine learning approaches that treat detection as a binary classification problem over URL and page-content features (Zamir et al., 2020; Ahammad et al., 2022). Classical models such as decision trees, SVMs, and random forests demonstrate strong performance on feature-engineered datasets; however, they often fail to generalize across data distributions (Jalil et al., 2022).

Deep learning approaches address this by learning representations directly from raw input. Maneriker et al. (2021) introduced URLTran, a Transformer-based model treating URLs as character sequences, which achieves improved detection on complex phishing URLs. Surveys by Tang and Mahmoud (2021) and Tamal et al. (2024) note that while deep learning yields higher accuracy, it carries greater computational cost and reduced interpretability compared to classical models.

Our project builds on this literature by comparing classical and neural approaches on the same large dataset, focusing on both predictive performance and feature interpretability, and explicitly investigating potential data leakage—an issue underexplored in prior work.

---

## 3. Dataset

We use the PhiUSIIL Phishing URL Dataset (UCI Machine Learning Repository, 2023) from the UC Irvine Machine Learning Repository. The dataset contains 235,795 labeled samples (0 = legitimate, 1 = phishing), with features extracted from URL strings and HTML page source. Non-numeric identifier columns (FILENAME, URL, Domain, Title, TLD) are dropped before model training. We identified 808 duplicate rows, noted as a potential source of inflated metrics. Features that were highly correlated to the label and highly correlated to each other were dropped. Labels correlated to each other included `NoOfLettersInUrl` and `DomainTitleMatchScore`. 25 of the features with the highest correlation to the labels were dropped. `URLSimilarityIndex` was noted to have the highest correlation with the labels at 0.86. All models use an 80/20 train/test split (random state=42), yielding 20,124 legitimate and 27,035 phishing samples in the test set.

---

## 4. Models

All models are implemented in Python using scikit-learn with `test_size=0.2` and `random_state=42`.

### Logistic Regression

After removing all features contributing to the data leakage, a Logistic Regression model was run through a k-fold cross-validation pipeline. `StandardScaler` was applied to the features. The regression model was run with the `lbfgs` solver and a cap of 1000 iterations (`max_iter=1000`). The k-fold had 5 splits with shuffling enabled. The model was fit to the training data through the pipeline and then made predictions on the test data. To assess the model, we computed a standard classification report with precision, recall, F1-score, and support.

### Decision Tree

A train split of the cleaned data was fed to an sklearn `DecisionTreeClassifier`. The classifier had a maximum depth of 2 (`max_depth=2`), a minimum number of samples required to split a node of 50 (`min_samples_split=50`), and a minimum of 25 samples per leaf (`min_samples_leaf=25`). The model made predictions on the test data, which were assessed with the standard classification report.

### Random Forest

The train split of the cleaned data was fed to a `RandomForestClassifier`. The classifier used 10 trees (`n_estimators=10`) with a maximum depth of 2 (`max_depth=2`), a minimum number of samples required to split a node of 50 (`min_samples_split=50`), and a minimum of 25 samples per leaf (`min_samples_leaf=25`). We set `max_features="sqrt"` to limit the number of features considered at each split. The model made predictions on the test data, which were assessed with the standard classification report.

### Neural Network

We implemented a feedforward multi-layer perceptron (MLP) classifier using `MLPClassifier` from scikit-learn. The features were transformed with `StandardScaler` prior to training the model. The first hidden layer had 32 neurons and the second had 16 (`hidden_layer_sizes=(32, 16)`). The maximum number of iterations for training was set to 500 (`max_iter=500`), the L2 regularization strength was set to 0.01 (`alpha=0.01`), and the initial learning rate was set to 0.001 (`learning_rate_init=0.001`). The model was set to stop training when the validation score plateaued (`early_stopping=True`), using 10% of the training data for validation (`validation_fraction=0.1`). The model made predictions on the test data, which were assessed with the standard classification report.

---

## 5. Results

### 5.1 Overall Performance

Table 1 reports performance for the four baselines on the held-out test set.

| Model | Acc. | Prec. | Rec. | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.86 | 0.86 | 0.86 | 0.86 |
| Decision Tree | 0.94 | 0.95 | 0.94 | 0.94 |
| Random Forest | 0.93 | 0.93 | 0.92 | 0.93 |
| Neural Network | 0.98 | 0.98 | 0.98 | 0.98 |

*Table 1: Macro-averaged performance on the 20% held-out test set.*

The Neural Network achieves the strongest overall score, with an accuracy of 0.98. The Decision Tree achieves 0.945 accuracy, and the Random Forest reaches 0.928. The Logistic Regression has the weakest accuracy score of 0.86.

### 5.2 Feature Performance Analysis

Table 2 reports the top 10 Random Forest Gini importances before the data was cleaned. `URLSimilarityIndex` ranks first at 19.5%; because it measures similarity to known legitimate URLs, it was confirmed to encode information correlated with label data and was a source of one of the leaks in the data. Table 3 reports the top 10 Logistic Regression features (before cleaning) by absolute coefficient magnitude. The LR rankings differ notably from RF: JavaScript count and special character ratios dominate, reflecting the linear model's reliance on scaled numerical contrast rather than tree-based splitting gain.

After the dataset was cleaned using the correlations between features, the feature importances were recalculated. Table 4 reports the top 10 Random Forest Gini importances after cleaning. The highest importance feature was `LargestLineLength`. Table 5 reports the top 10 Logistic Regression features by absolute coefficient magnitude after cleaning, indicating which numeric features have the greatest impact on the classification output. The largest feature was `NoOfDegitsInURL`, which is notably also the second most important feature by Gini importance.

| Feature | Importance |
|---|---|
| URLSimilarityIndex | 0.1953 |
| NoOfExternalRef | 0.1700 |
| LineOfCode | 0.1338 |
| NoOfImage | 0.1133 |
| NoOfSelfRef | 0.0864 |
| NoOfJS | 0.0747 |
| HasSocialNet | 0.0354 |
| NoOfCSS | 0.0296 |
| HasDescription | 0.0240 |
| HasCopyrightInfo | 0.0223 |

*Table 2: Top 10 Random Forest feature importances (Gini-based).*

| Feature | \|β\| |
|---|---|
| NoOfJS | 8.516 |
| NoOfOtherSpecialCharsInURL | 3.271 |
| LetterRatioInURL | 2.925 |
| NoOfCSS | 2.894 |
| SpacialCharRatioInURL | 2.879 |
| DigitRatioInURL | 2.763 |
| NoOfDegitsInURL | 2.424 |
| HasSocialNet | 2.283 |
| NoOfEmptyRef | 1.868 |
| NoOfQMarkInURL | 1.591 |

*Table 3: Top 10 LR features by absolute coefficient magnitude (41-feature pruned set, standardized inputs).*

| Feature | Importance |
|---|---|
| LargestLineLength | 0.3134 |
| NoOfDegitsInURL | 0.2677 |
| NoOfCSS | 0.2023 |
| URLLength | 0.0869 |
| TLDLegitimateProb | 0.0661 |
| NoOfSubDomain | 0.0282 |
| NoOfPopup | 0.0255 |
| NoOfQMarkInURL | 0.0078 |
| NoOfEmptyRef | 0.0021 |
| HasExternalFormSubmit | 0.0001 |

*Table 4: Top 10 Random Forest feature importances (Gini-based) after cleaning.*

| Feature | \|β\| |
|---|---|
| NoOfDegitsInURL | 13.8350 |
| NoOfEmptyRef | 12.7020 |
| NoOfiFrame | 9.5368 |
| NoOfPopup | 5.3132 |
| URLLength | 4.1733 |
| NoOfQMarkInURL | 2.5633 |
| HasObfuscation | 0.8796 |
| HasExternalFormSubmit | 0.6587 |
| NoOfEqualsInURL | 0.6349 |
| IsDomainIP | 0.5486 |

*Table 5: Top 10 LR features by absolute coefficient magnitude (22-feature pruned set, standardized inputs) after cleaning.*

### 5.3 Ablation Study

We ran 5-fold cross-validation on the Random Forest under three feature configurations. With all features, mean CV accuracy is 1.0000. Removing only `URLSimilarityIndex` yields 0.9999. Removing the top five features (`URLSimilarityIndex`, `NoOfExternalRef`, `LineOfCode`, `NoOfImage`, `NoOfSelfRef`) yields 0.9995. The marginal impact of each removal suggests a strong predictive signal is distributed across the feature set. After further investigation and removal of leaky features, the Random Forest accuracy was reduced to 0.93.

---

## 6. Discussion

In our earlier analysis of the project, our models performed with perfect accuracy. After we removed possible feature leakage, the performance of the models became more balanced. Logistic Regression dropped to a 0.86 accuracy, while the Decision Tree and Random Forest achieved about 0.945 and 0.928. This suggests that the earlier model performance was inflated by features that simplified the classification too much.

The Neural Network achieved the strongest overall result, with an overall accuracy of 0.98 on the test set. This shows that the dataset still contained strong predictive ability after removing features with label correlation. The Logistic Regression model likely did poorly compared to the other models due to the model being better suited for capturing linear relationships in the data. The Decision Tree and Random Forest are not linear models, so they can capture the non-linearity of the phishing data.

Doing more preprocessing on the data had a major impact on our final results. We removed many features that we suspected of causing data leakage, which made a drastic change in our models. This slightly lowered our model accuracy, but produced results that better reflect real-world performance.

---

## 7. Future Works

The prevalence and complexity of phishing attacks allow our models to continuously have room for improvement. There are a number of changes that can be made to each model that might improve their computation costs, performance, or both.

For further improvement on the Logistic Regression model, further feature engineering and polynomial expansion could be explored. From the coefficient analysis, we learned that the regression model heavily relies on numerical contrast. From this, we could create more features like ratios between the HTML features and URL features to give the linear model more to work with, though this requires careful validation to ensure the new features do not introduce additional data leakage similar to what was identified earlier in our study. Similarly, the model struggles with the non-linear relationships coming from the data. Adding interaction terms or 2nd degree features could improve the model's performance around complex boundaries. However, this comes at the cost of a significantly larger feature space, which could increase computation time and risk overfitting without proper regularization tuning. Despite these tradeoffs, these improvements are worth exploring given that Logistic Regression is the weakest performing model in our study at 0.86 accuracy, leaving the most room for gain among the four models.

Multiple pieces of the Decision Tree architecture can be tuned. Hyperparameter tuning can be performed on the depth, split criteria, and leaf size of the model with a CV grid search. While this has the potential to meaningfully improve upon the current 0.945 accuracy, an exhaustive grid search over many hyperparameter combinations can be computationally expensive, so limiting the search space to reasonable ranges based on the dataset size would be important. We can also implement cost-complexity pruning by using a deeper model and increasing `ccp_alpha` to prune more nodes. This approach is preferable to pre-pruning via depth limits because it allows the tree to first capture complex patterns before selectively removing the weakest branches, leading to a better bias-variance tradeoff. However, selecting the optimal `ccp_alpha` value requires additional cross-validation, and setting it too high risks oversimplifying the tree and discarding genuinely useful splits, while setting it too low may leave the tree overfitted to the training data. Despite these challenges, the Decision Tree remains an attractive model to improve due to its human-readable rules that could be directly useful for security analysts trying to understand why a particular URL was flagged as phishing.

To further improve the Random Forest, similar to the Decision Tree, depth can be tuned. Allowing the trees to grow deeper would enable the model to capture more complex patterns in the phishing data, though this must be balanced carefully as unconstrained depth can lead to overfitting and significantly increased memory usage. Likewise, the model could benefit from the number of trees being increased. The current forest uses only 10 estimators, which is far below the typical range of 100–500 used in practice, and may contribute to its lower accuracy of 0.928 compared to the Neural Network's 0.98. Increasing the number of estimators would likely reduce variance and improve the accuracy score; however, this comes at a linear increase in both training time and memory consumption, which should be considered if the model is intended for real-time phishing detection where speed is critical. We also could implement out-of-bag evaluation for a better estimate of the accuracy of the model without the cost of cross-validation. This is particularly useful for the Random Forest since OOB evaluation is a natural byproduct of the bootstrapping process, adding no meaningful computational overhead. However, it is worth noting that OOB estimates can be slightly pessimistic compared to cross-validation, since each tree is only evaluated on roughly one third of the training data, and so cross-validation may still be preferable when a highly precise accuracy estimate is needed.

To improve the Neural Network, we could explore changing the architecture of the model or using more complex systems. The Neural Network could be simplified to improve computation costs, which would be beneficial if the model were to be deployed in a real-time phishing detection system where low latency is critical. However, it is already quite shallow with hidden layers of only (32, 16), meaning further simplification risks underfitting and degrading the already strong 0.98 accuracy. We could consider a network that is deeper or wider with dropout to improve the accuracy, though this introduces a larger hyperparameter search space covering layer count, layer width, and dropout rate, which would require careful tuning to avoid overfitting. Despite this cost, a deeper architecture may be able to capture more subtle patterns in the feature data that the current shallow network misses. We could also use a more complex system entirely. Instead of depending on the features from the dataset, we could use a Transformer encoder to learn directly from the URL string data. This would remove the reliance on hand-crafted features entirely, making the model more robust to feature leakage and more adaptable to new phishing strategies that may not be captured by the current feature set. However, Transformer-based models carry significantly greater computational cost in both training and inference, and would require significantly more labeled data than is currently available to reach their full potential.

---

## 8. Conclusion

We have presented four models for detecting phishing URLs: a Logistic Regression model, a Decision Tree, a Random Forest, and a two-layer Neural Network. Each model presents its own unique strengths and weaknesses when deployed in a real URL detection setting, and each contributes meaningfully to our understanding of the phishing classification problem.

The most significant finding of our work has been the detection and management of leaky features. Early forms of our models achieved perfect accuracy because of features that would not realistically be available in real-world deployment, causing the models to perform exceptionally well in training and testing to an unreasonable extent. The leaky data allowed the models to essentially "cheat" at detecting phishing URLs, the most guilty feature being `URLSimilarityIndex`, which measures how similar a URL is to known legitimate URLs. The detection and removal of features like these was crucial to the development of realistic and trustworthy models, and highlights the importance of rigorous feature auditing as a step in any machine learning pipeline handling sensitive classification tasks.

Though each model has its own benefits and drawbacks, two of the models are the most generally effective and suitable for real-world applications. The Neural Network achieved the strongest result at an accuracy of 0.98, and at only two layers, the computational cost is reasonable relative to its performance. The Decision Tree and Random Forest are also well suited for this problem, achieving accuracy scores of 0.94 and 0.93 respectively. These models produce human-readable outputs that would be practical for real-world deployment, particularly in settings where security analysts need to understand and justify individual classification decisions. The weakest model, the Logistic Regression, achieved an accuracy score of 0.86, reflecting its fundamental limitation with the non-linear nature of the data.

Performance is not the only important output of these models. Interpretability is also a critical factor in real-world applications. However, we cannot make strong deployment claims with full confidence given several limitations of the dataset: 808 duplicate rows, heavy reliance on hand-crafted features many of which were found to be correlated or label-adjacent, and no analysis of the variety and distribution of the URLs themselves. Therefore, further validation on independent, real-world data is required before any of our models could be considered deployment-ready.

As discussed in the future works section, the Neural Network's strong baseline performance suggests that exploring Transformer encoder architectures that learn directly from raw URL strings is a promising direction. This research has established a realistic and effective foundation for further work in URL phishing classification, contributing a more grounded and reproducible baseline to the broader field of phishing detection.

---

## References

Ahammad, S. H., Kale, S. D., Upadhye, G. D., Pande, S. D., Babu, E. V., Dhumane, A. V., and Bahadur, D. K. J. 2022. Phishing URL detection using machine learning methods. *Advances in Engineering Software*, 173:103288.

Jalil, S., Usman, M., and Fong, A. 2022. Highly accurate phishing URL detection based on machine learning. *Journal of Ambient Intelligence and Humanized Computing*.

Maneriker, P., Stokes, J. W., Lazo, E. G., Carutasu, D., Tajaddodianfar, F., and Gururajan, A. 2021. URLTran: Improving phishing URL detection using transformers. *arXiv preprint arXiv:2106.05256*.

Tamal, M. A., Islam, M. K., Bhuiyan, T., and Sattar, A. 2024. Dataset of suspicious phishing URL detection. *Frontiers in Computer Science*, 6.

Tang, L. and Mahmoud, Q. H. 2021. A survey of machine learning-based solutions for phishing website detection. *Machine Learning and Knowledge Extraction*, 3(3):672–694.

Zamir, A., Khan, H. U., Iqbal, T., Yousaf, N., Aslam, F., Anjum, A., and Hamdani, M. 2020. Phishing web site detection using diverse machine learning algorithms. *The Electronic Library*, 38(1):65–80. <br>
phishing phishing phsishing pshishng pigsh ighs phsihgping

![A description of the image](cute_cat.jpg)

