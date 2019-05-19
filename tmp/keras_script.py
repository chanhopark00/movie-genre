import numpy as np
from keras.models import Sequential
from my_classes import DataGenerator

# Parameters
params = {'dim': (278,185,3),
          'batch_size': 30,
          'n_classes': 20,
          'n_channels': 1,
          'shuffle': True}

# Datasets
partition = {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}

labels = # Labels

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

# Design model
model = Sequential()
[...] # Architecture
model.compile()

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)