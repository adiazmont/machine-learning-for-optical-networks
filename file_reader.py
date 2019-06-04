import csv
import numpy as np

class FileReader():
    def read_array_two_class(self, _file):
        # Retrieve and format data for training set
        with open(_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            samples = []
            osnr_labels = []
            for row in csv_reader:
                samples.append([float(i) for i in row[:-1]])
                osnr_labels.append(float(row[-1]))
            csv_file.close()
            
        labels = []
        for osnr in osnr_labels:
            if osnr < 17:
                labels.append([1, -1])
            else:
                labels.append([-1, 1])

        X = np.asarray(samples)
        y = np.asarray(labels)

        return X, y
        
    def read_array_three_class(self, _file):
        # Retrieve and format data for training set
        with open(_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            samples = []
            osnr_labels = []
            for row in csv_reader:
                samples.append([float(i) for i in row[:-1]])
                osnr_labels.append(float(row[-1]))
            csv_file.close()
            
        labels = []
        classA = 0
        classB = 0
        classC = 0
        classD = 0
        for osnr in osnr_labels:
            if osnr >= 17:
                labels.append([1, 0, 0, 0])
                classA += 1
            elif osnr >= 14:
                labels.append([0, 1, 0, 0])
                classB += 1
            elif osnr >= 10:
                labels.append([0, 0, 1, 0])
                classC += 1
            else:
                labels.append([0, 0, 0, 1])
                classD += 1
        total_samples = classA + classB + classC + classD
        print("Class A: %s samples \nClass B: %s samples \nClass C: %s samples \nClass D: %s samples \nTotal samples: %s" %(classA, classB, classC, classD, total_samples))

        X = np.asarray(samples)
        y = np.asarray(labels)

        return X, y
