These scripts can be used to reproduce the results obtained in our analysis. 

feature_extractor.py can be used to extract features from the BAM file and the VCF generated. It takes the reference file as an argument. The output here is a CSV file with all the data.

build_dataset.py takes the bcftools/freebayes obtained VCF and labels them using the truth set.

train.py takes the data set and trains a gradient boosted machine.

predict_dira.py finally takes the model and the original VCF as an input and outputs a new VCF.
