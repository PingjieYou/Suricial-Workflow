import pickle
import configs
import numpy as np

with open("." + configs.NumPicklePath, "rb") as f:
    num_list = pickle.load(f)

with open("." + configs.ImagePicklePath, "rb") as f:
    img_list = pickle.load(f)

with open("." + configs.ToolPicklePath, "rb") as f:
    tool_list = pickle.load(f)

with open("." + configs.PhasePicklePath, "rb") as f:
    phase_list = pickle.load(f)

with open("." + configs.AnticipationPicklePath, "rb") as f:
    anticipation_list = pickle.load(f)

# cholec80==================
train_file_paths_80 = []
test_file_paths_80 = []
val_file_paths_80 = []
val_labels_80 = []
train_labels_80 = []
test_labels_80 = []

train_num_each_80 = []
val_num_each_80 = []
test_num_each_80 = []

print(phase_list[156])
assert 0

stat = np.zeros(7).astype(int)
for i in range(40):
    train_num_each_80.append(num_list[i]) # 数据的数量
    for j in range(num_list[i]):
        train_file_paths_80.append(img_list[j])
        train_labels_80.append(anticipation_list[j])
    print(np.max(train_labels_80))
    assert 0

print(len(train_file_paths_80))
print(len(train_labels_80))
print(np.max(np.array(train_labels_80)[:, 0]))
print(np.min(np.array(train_labels_80)[:, 0]))

for i in range(40, 48):
    val_num_each_80.append(num_list[i])
    for j in range(num_list[i]):
        val_file_paths_80.append(img_list[j])
        val_labels_80.append(anticipation_list[j])

for i in range(40, 80):
    test_num_each_80.append(num_list[i])
    for j in range(num_list[i]):
        test_file_paths_80.append(img_list[j])
        test_labels_80.append(anticipation_list[j])

print(len(val_file_paths_80))
print(len(val_labels_80))

# cholec80==================


train_val_test_paths_labels = []
# train_val_test_paths_labels.append(train_file_paths_19)
train_val_test_paths_labels.append(train_file_paths_80)
train_val_test_paths_labels.append(val_file_paths_80)

# train_val_test_paths_labels.append(train_labels_19)
train_val_test_paths_labels.append(train_labels_80)
train_val_test_paths_labels.append(val_labels_80)

# train_val_test_paths_labels.append(train_num_each_19)
train_val_test_paths_labels.append(train_num_each_80)
train_val_test_paths_labels.append(val_num_each_80)

train_val_test_paths_labels.append(test_file_paths_80)
train_val_test_paths_labels.append(test_labels_80)
train_val_test_paths_labels.append(test_num_each_80)

with open('train_val_paths_labels1.pkl', 'wb') as f:
    pickle.dump(train_val_test_paths_labels, f)

print('Done')
