# Prediction for the computing resource consumption of VM I/O thread
This is our implementation for the paper:

Jiyou Lee, Kwanhoon Lee, and Chuck Yoo (2021). [A Machine-Learning-Based Approach towards Modeling the CPU Quota Requirements of Virtual Machine I/O Threads](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10530084) 

# Abstract
With cloud computing gaining popularity, providing reliable network performance within virtualized environments has become a major issue. Previous research states that the resource allocation of I/O threads (i.e., kernel threads that manage the I/O operations of a virtual machine) plays an important role in determining network performance. Therefore, in this paper, we aim to predict the amount of CPU bandwidth I/O threads utilize in various different networking situations by adopting a machine learning approach. Specifically, we experiment with building efficient machine learning models that estimate the CPU quota requirements of the I/O threads for virtual machines with specific network performance needs.

# Requirements
1. Python `3.5 ~ 3.7`
2. sklearn `0.20.0`
3. pandas `0.24.0`
4. numpy `1.16.0`
</br>

# Example to run the codes
```bash
# to run linear regress for prediction
python3 multivariate_linear_regression.py

# to run random_forest_regression for prediction
python3 multivariate_random_forest_regression.py

# to run support_vector_regression for prediction
python3 multivariate_vector_regression.py
```

