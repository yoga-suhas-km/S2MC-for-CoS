# A Novel Statistical Segmented Approach for Class-of-Service Network Traffic Classification

Please download the datasets from below URL

**URL for Dataset I:** https://ieee-dataport.org/documents/network-service-traffic

**URL for Dataset II:** https://www.unb.ca/cic/datasets/vpn.html

Extract the csv from pcap that includes packet length, inter-arrival time, packet direction and timestamps.
Place the CSVs of DatasetI in a directory named "IEEE_dataport_pre_processed" and CSVs of DatasetII in a
directory names "UNB_pre_processed".

Dependent packages: sklearn, pandas, keras, numpy, matplot, seaborn, joblib.

Please execute rf.py as shown in the below commands

1) To run S2MC for Dataset I
* $ python rf.py dataset1

2) To run S2MC for Dataset II
* $ python rf.py dataset2

3) To plot histograms for sliced data of Dataset I
* $ python rf.py plot dataset1

4)  To plot histograms for sliced data of Dataset II
* $ python rf.py plot dataset2
