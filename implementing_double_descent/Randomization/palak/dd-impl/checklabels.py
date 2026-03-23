import os
import pickle

from cifar10_utils import load_cifar10_batch


def load_all_train_labels(dataset_folder_path):
	labels = []
	for batch_id in range(1, 6):
		_, batch_labels = load_cifar10_batch(dataset_folder_path, batch_id)
		labels.extend(batch_labels)
	return labels


def load_test_labels(dataset_folder_path):
	test_batch_path = os.path.join(dataset_folder_path, "test_batch")
	with open(test_batch_path, mode="rb") as file:
		batch = pickle.load(file, encoding="latin1")
	return batch["labels"]


def main():
	dataset_folder_path = "data/cifar-10-batches-py"

	train_labels = load_all_train_labels(dataset_folder_path)
	test_labels = load_test_labels(dataset_folder_path)
	all_labels = train_labels + test_labels

	print("Train label set:", set(train_labels))
	print("Test label set:", set(test_labels))
	print("All label set:", set(all_labels))
	


if __name__ == "__main__":
	main()

