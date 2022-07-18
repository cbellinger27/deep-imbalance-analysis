This repository includes the code for  manuscript, "The Class Imbalance Problem in Deep Learning", by Kushankur Ghosh ,  Colin Bellinger,  Roberto Corizzo,  Paula Branco,  Bartosz Krawczyk and Nathalie Japkowicz submitted to Machine Learning

This code correspondes to the paper entitled 'The Class Imbalance Problem in Deep Learning' submitted to the Special Issue on Imbalanced Learning in the journal Machine Learning with co-authorship from:

Roberto Corizzo - rcorizzo@american.edu Nathalie Japkowicz - japkowic@american.edu Kushankur Ghosh - kushanku@ualberta.ca Paula Branco - pbranco@uottawa.ca Bartosz Krawczyk - bkrawczyk@vcu.edu

The objective of the paper is to asses the affectiveness of depth at mitigating bias due to class imbalance, and to how this is impacted by problem complexity.

EXPERIMENTS
-------------------------------------------
-------------------------------------------
Data
-------------------------------------------

Backbone datasets: synthetic tabular dataset that corresponds to the imbalance analysis originally undertaken in  in Japkowicz
and Stephen (2002)
Image datasets: imbalanced versions of CIFAR-10 and MNIST-Fashion, plus imbalanced versions of Shapes dataset proposed by (El Korchi and Ghanou in 2020)
Text datasets: imbalanced versions of 20NewsGroup (Alhenaki and Hosny, 2019) and Job Classification (https://www.kaggle.com/adarshsng/predicting-job-type-category-by-job-description?select=train.csv)

The datasets are not included in this repository due to storage limitations but are available upon request

Deep networks
-------------------------------------------
MLP: Deep versions of the MLP (fully connected deep network) were applied to the backbone and text datasets. The text datasets were pre-processed with TF-IDF prior to input in the network 
CNN: Standard CNN architectures were used for the text data

The code for the deep learning models is contained in src/models/

Results
-------------------------------------------
Performance is assessed based on the average g-mean