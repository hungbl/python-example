# Load CSV using Pandas from URL
import pandas
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
print(data.shape)
# Load CSV using csv
# import csv
# file = "2018JulyChargebackReport.csv"
# with open(file) as csvfile:
#     reader = csv.reader(csvfile, delimiter=',')
#     for row in reader:
#         print(', '.join(row))

