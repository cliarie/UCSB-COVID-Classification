# UCSB-COVID-Classification

Previous COVID-19 research has observed lung lesions of COVID-19 are generally peripherally distributed and bilateral, but other viral pneumonia lesions are not subject to those standards. Since these factors distinguishing the pneumonias are location based, we propose a novel method of COVID-19 classification by using segmented overlays as part of the classification process. 

For our segmentation overlay method, we initially tried it on a ResNet34 binary classifier of COVID-19 and PNA. This approach improved the accuracy by 1.54%,  producing a 0.7439 overall accuracy. Extending the method to ResNet152 multi-class classifier, it improved ResNet152 accuracy by 6.74% with an overall accuracy of 0.7790 and improved upon the baseline ResNet50 model by 9.23%. This approach generated 82% subject-wise accuracy, correctly diagnosing 44 out of the 50 test scans.
