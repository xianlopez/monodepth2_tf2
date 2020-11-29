import tensorflow as tf
from model import training_model
from loss import compute_loss


model = training_model()
optimizer = tf.optimizers.Adam(learning_rate=1e-4)
model.compile(loss=compute_loss, optimizer=optimizer)

# model.fit(train_dataset, validation_data=val_dataset, epochs=20)
