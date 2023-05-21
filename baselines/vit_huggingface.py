from datasets import load_dataset
from transformers import pipeline
dataset = load_dataset('cifar10')

# model = AutoModelForImageClassification.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

correct = 0
total = 0

classifier = pipeline('image-classification', model='aaraki/vit-base-patch16-224-in21k-finetuned-cifar10')
print("len(dataset['test']) ==", len(dataset['test']))
for example in dataset["test"]:
    print("example ==", example)
    image = example["img"]
    result = classifier(image)
    label = result[0]["label"]
    if (label == labels[example['label']]):
        print("yes!!!")
        correct += 1
    total += 1
    confidence = result[0]["score"]
    print(f"Predicted label: {label}, Confidence {confidence}, Actual label: {labels[example['label']]}")
    if total >= 100:
        break
print("accuracy ==", correct/total)