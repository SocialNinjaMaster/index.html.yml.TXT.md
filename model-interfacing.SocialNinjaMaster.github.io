model_id = "ibm/granite-13b-chat-v2"
parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 200,
    "repetition_penalty": 1
project_id = os.getenv("PROJECT_ID")
space_id = os.getenv("SPACE_ID")
Jupiter-Notebook
from ibm_watsonx_ai.foundation_models import Model

model = Model(
	model_id = model_id,
	params = parameters,
	credentials = get_credentials(),
	project_id = project_id,
	space_id = space_id
	)
prompt_input = """```python
import tensorflow as tf

# Load your data
train_data = ...
test_data = ...

# Compile your model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile your model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train your model
model.fit(train_data, epochs=10)

# Evaluate your model
test_loss, test_acc = model.evaluate(test_data)
print(f'Test accuracy: {test_acc:.2f}')
```

In this example, we define a simple TensorFlow model with one input layer of shape `(784,)` and two output layers of shape `(None, 10)`. We then compile the model with the `adam` optimizer, `sparse_categorical_crossentropy` loss function, and `accuracy` metric.

Next, we train the model using the `fit` method, passing in the training data and specifying the number of epochs to train for.

Finally, we evaluate the model on the test data using the `evaluate` method, which returns the test loss and accuracy.

It's important to note that the `evaluate` method should only be used for evaluation and not for training, as it uses the `train_data` instead of the `test_data`. This ensures that the model's performance on the test data is not influenced by the training process.

Additionally, when evaluating the model,"""print("Submitting generation request...")
generated_response = model.generate_text(prompt=prompt_input, guardrails=True)
print(generated_response)print""";
//
///
}
