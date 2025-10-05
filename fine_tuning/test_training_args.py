from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./output",
    evaluation_strategy="steps",
    per_device_train_batch_size=1,
    num_train_epochs=1
)

print(args)