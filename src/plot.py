import matplotlib.pyplot as plt
import numpy as np

# Assuming you have your metrics stored in dictionaries like this:
model1_metrics = {'accuracy': 0.85, 'precision': 0.88, 'recall': 0.82, 'f1': 0.85, 'mcc': 0.84}
model2_metrics = {'accuracy': 0.89, 'precision': 0.91, 'recall': 0.87, 'f1': 0.89, 'mcc': 0.88}
model3_metrics = {'accuracy': 0.92, 'precision': 0.93, 'recall': 0.91, 'f1': 0.92, 'mcc': 0.91}

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Plot data
ax.plot(model1_metrics.keys(), model1_metrics.values(), label='Model 1')
ax.plot(model2_metrics.keys(), model2_metrics.values(), label='Model 2')
ax.plot(model3_metrics.keys(), model3_metrics.values(), label='Model 3')

# Add a legend
ax.legend()

# Show the plot
plt.show()


# Create a figure and a set of subplots
fig, axs = plt.subplots(2)

# Plot line graph
axs[0].plot(model1_metrics.keys(), model1_metrics.values(), label='Model 1')
axs[0].plot(model2_metrics.keys(), model2_metrics.values(), label='Model 2')
axs[0].plot(model3_metrics.keys(), model3_metrics.values(), label='Model 3')
axs[0].legend()
axs[0].set_title('Line Graph')

# Prepare data for bar chart
labels = model1_metrics.keys()
model1_values = model1_metrics.values()
model2_values = model2_metrics.values()
model3_values = model3_metrics.values()

x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars

# Plot bar chart
rects1 = axs[1].bar(x - width, model1_values, width, label='Model 1')
rects2 = axs[1].bar(x, model2_values, width, label='Model 2')
rects3 = axs[1].bar(x + width, model3_values, width, label='Model 3')

axs[1].set_title('Bar Chart')
axs[1].set_xticks(x)
axs[1].set_xticklabels(labels)
axs[1].legend()

fig.tight_layout()

plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Assuming you have your metrics stored in dictionaries like this:
model1_metrics = {'Class 1': 0.85, 'Class 2': 0.88, 'Class 3': 0.82, 'Class 4': 0.85, 'Class 5': 0.84, 'Class 6': 0.88, 'Class 7': 0.82}
model2_metrics = {'Class 1': 0.89, 'Class 2': 0.91, 'Class 3': 0.87, 'Class 4': 0.89, 'Class 5': 0.88, 'Class 6': 0.91, 'Class 7': 0.87}
model3_metrics = {'Class 1': 0.92, 'Class 2': 0.93, 'Class 3': 0.91, 'Class 4': 0.92, 'Class 5': 0.91, 'Class 6': 0.93, 'Class 7': 0.91}

labels = model1_metrics.keys()
model1_values = list(model1_metrics.values())
model2_values = list(model2_metrics.values())
model3_values = list(model3_metrics.values())

x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots()

# Add bars for each model
rects1 = ax.bar(x - width, model1_values, width, label='Model 1')
rects2 = ax.bar(x, model2_values, width, label='Model 2')
rects3 = ax.bar(x + width, model3_values, width, label='Model 3')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Classes')
ax.set_ylabel('Scores')
ax.set_title('Scores by class and model')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()


import matplotlib.pyplot as plt

# Assuming you have your accuracy stored in a dictionary like this:
accuracy = {'Model 1': 0.85, 'Model 2': 0.88, 'Model 3': 0.82}

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Plot data
ax.plot(accuracy.keys(), accuracy.values(), marker='o')

# Add labels and title
ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy of different models')

# Annotate each point with the model name
for i, txt in enumerate(accuracy.keys()):
    ax.annotate(txt, (i, list(accuracy.values())[i]))

# Show the plot
plt.show()

import matplotlib.pyplot as plt

# Assuming you have your metrics stored in dictionaries like this:
model1_metrics = {'accuracy': 0.85, 'precision': 0.88, 'recall': 0.82}
model2_metrics = {'accuracy': 0.89, 'precision': 0.91, 'recall': 0.87}
model3_metrics = {'accuracy': 0.92, 'precision': 0.93, 'recall': 0.91}

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Plot data
ax.plot(model1_metrics.keys(), model1_metrics.values(), marker='o', label='Model 1')
ax.plot(model2_metrics.keys(), model2_metrics.values(), marker='o', label='Model 2')
ax.plot(model3_metrics.keys(), model3_metrics.values(), marker='o', label='Model 3')

# Add labels and title
ax.set_xlabel('Evaluation parameters')
ax.set_ylabel('Scores')
ax.set_title('Performance of different models on various metrics')

# Add a legend
ax.legend()

# Show the plot
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Assuming you have your metrics stored in dictionaries like this:
model1_metrics = {'accuracy': 0.85, 'macro-precision': 0.88, 'recall': 0.82}
model2_metrics = {'accuracy': 0.89, 'macro-precision': 0.91, 'recall': 0.87}
model3_metrics = {'accuracy': 0.92, 'precision': 0.93, 'recall': 0.91}

labels = model1_metrics.keys()
model1_values = list(model1_metrics.values())
model2_values = list(model2_metrics.values())
model3_values = list(model3_metrics.values())

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars
space = 0.1  # the space between the bars

fig, ax = plt.subplots()

# Add bars for each model
rects1 = ax.bar(x - width - space/2, model1_values, width, label='Model 1')
rects2 = ax.bar(x, model2_values, width, label='Model 2')
rects3 = ax.bar(x + width + space/2, model3_values, width, label='Model 3')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Scores by metric and model')
ax.set_xticks(x)
ax.set_xticklabels(labels)

# Move the legend below the x-axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

# Function to add labels on top of the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# Add more space between the bars
plt.subplots_adjust(bottom=0.2)

fig.tight_layout()

plt.show()

import matplotlib.pyplot as plt

# Assuming you have your class-wise metrics stored in dictionaries like this:
model1_metrics = {'Class 1': 0.85, 'Class 2': 0.88, 'Class 3': 0.82, 'Class 4': 0.85, 'Class 5': 0.84, 'Class 6': 0.88, 'Class 7': 0.82}
model2_metrics = {'Class 1': 0.89, 'Class 2': 0.91, 'Class 3': 0.87, 'Class 4': 0.89, 'Class 5': 0.88, 'Class 6': 0.91, 'Class 7': 0.87}
model3_metrics = {'Class 1': 0.92, 'Class 2': 0.93, 'Class 3': 0.91, 'Class 4': 0.92, 'Class 5': 0.91, 'Class 6': 0.93, 'Class 7': 0.91}

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Plot data
ax.plot(model1_metrics.keys(), model1_metrics.values(), marker='o', label='Model 1')
ax.plot(model2_metrics.keys(), model2_metrics.values(), marker='o', label='Model 2')
ax.plot(model3_metrics.keys(), model3_metrics.values(), marker='o', label='Model 3')

# Add labels and title
ax.set_xlabel('Classes')
ax.set_ylabel('Scores')
ax.set_title('Class-wise performance of different models')

# Add a legend
ax.legend()

# Show the plot
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Assuming you have your class-wise metrics stored in dictionaries like this:
metrics = ['accuracy', 'precision', 'recall', 'f1 score', 'mcc']
class1_metrics = [0.85, 0.88, 0.82, 0.83, 0.84]
class2_metrics = [0.89, 0.91, 0.87, 0.88, 0.89]
class3_metrics = [0.92, 0.93, 0.91, 0.92, 0.93]

x = np.arange(len(metrics))  # the label locations
width = 0.2  # the width of the bars
space = 0.05  # the space between the bars

fig, ax = plt.subplots()

# Add bars for each class
rects1 = ax.bar(x - width - space, class1_metrics, width, label='Class 1')
rects2 = ax.bar(x, class2_metrics, width, label='Class 2')
rects3 = ax.bar(x + width + space, class3_metrics, width, label='Class 3')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Metric-wise performance of different classes')
ax.set_xticks(x)
ax.set_xticklabels(metrics)

# Move the legend below the x-axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

# Function to add labels on top of the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# Add more space between the bars
plt.subplots_adjust(bottom=0.2)

fig.tight_layout()

plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have your class-wise metrics stored in dictionaries like this:
metrics = ['accuracy', 'precision', 'recall', 'f1 score', 'mcc']
model1_class1_metrics = [0.85, 0.88, 0.82, 0.83, 0.84]
model1_class2_metrics = [0.89, 0.91, 0.87, 0.88, 0.89]
model1_class3_metrics = [0.92, 0.93, 0.91, 0.92, 0.93]

model2_class1_metrics = [0.86, 0.89, 0.83, 0.84, 0.85]
model2_class2_metrics = [0.90, 0.92, 0.88, 0.89, 0.90]
model2_class3_metrics = [0.93, 0.94, 0.92, 0.93, 0.94]

x = np.arange(len(metrics))  # the label locations
width = 0.2  # the width of the bars
space = 0.05  # the space between the bars

fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Function to add labels on top of the bars
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

for i, ax in enumerate(axs):
    # Add bars for each class
    rects1 = ax.bar(x - width - space, eval(f'model{i+1}_class1_metrics'), width, label='Class 1')
    rects2 = ax.bar(x, eval(f'model{i+1}_class2_metrics'), width, label='Class 2')
    rects3 = ax.bar(x + width + space, eval(f'model{i+1}_class3_metrics'), width, label='Class 3')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title(f'Metric-wise performance of different classes for Model {i+1}')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)

    # Move the legend below the x-axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)

# Add more space between the bars
plt.subplots_adjust(bottom=0.2, hspace=0.5)

fig.tight_layout()

plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have your class-wise metrics stored in dictionaries like this:
classes = ['Class 1', 'Class 2', 'Class 3']
metrics = ['accuracy', 'precision', 'recall', 'f1 score', 'mcc']

model1_metrics = [[0.85, 0.88, 0.82, 0.83, 0.84],
                  [0.89, 0.91, 0.87, 0.88, 0.89],
                  [0.92, 0.93, 0.91, 0.92, 0.93]]

model2_metrics = [[0.86, 0.89, 0.83, 0.84, 0.85],
                  [0.90, 0.92, 0.88, 0.89, 0.90],
                  [0.93, 0.94, 0.92, 0.93, 0.94]]

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(model1_metrics, annot=True, ax=axs[0], xticklabels=metrics, yticklabels=classes, cmap="YlGnBu")
axs[0].set_title('Model 1')

sns.heatmap(model2_metrics, annot=True, ax=axs[1], xticklabels=metrics, yticklabels=classes, cmap="YlGnBu")
axs[1].set_title('Model 2')

plt.show()


import matplotlib.pyplot as plt

# Assuming you have accuracy scores from multiple runs of each model
model1_accuracies = [0.85, 0.86, 0.84, 0.85, 0.87]
model2_accuracies = [0.89, 0.88, 0.90, 0.91, 0.89]
model3_accuracies = [0.92, 0.93, 0.91, 0.92, 0.93]

data = [model1_accuracies, model2_accuracies, model3_accuracies]

fig, ax = plt.subplots()

# Create the box plot
ax.boxplot(data)

# Set the x-axis labels
ax.set_xticklabels(['Model 1', 'Model 2', 'Model 3'])

# Add labels and title
ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')
ax.set_title('Distribution of accuracy scores for each model')

plt.show()


import matplotlib.pyplot as plt

# Assuming you have accuracy and precision for each class
accuracy = [0.85, 0.89, 0.92]
precision = [0.88, 0.91, 0.93]
classes = ['Class 1', 'Class 2', 'Class 3']

# Create scatter plot
plt.scatter(accuracy, precision)

# Add labels and title
plt.xlabel('Accuracy')
plt.ylabel('Precision')
plt.title('Scatter plot of accuracy vs precision for each class')

# Label each point
for i in range(len(classes)):
    plt.annotate(classes[i], (accuracy[i], precision[i]))

plt.show()

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have accuracy, precision, recall, and f1-score for each class
data = {
    'accuracy': [0.85, 0.89, 0.92],
    'precision': [0.88, 0.91, 0.93],
    'recall': [0.82, 0.87, 0.31],
    'f1_score': [0.83, 0.88, 0.92]
}
df = pd.DataFrame(data, index=['Model 1', 'Model 2', 'Model 3'])


# Create correlation matrix
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)

plt.show()

