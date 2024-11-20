from simpletransformers.classification import ClassificationModel
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_recall_fscore_support as prfs
import csv
import pandas as pd
from functools import partial

# load data
train=pd.read_csv("train.csv", header=0, names=["labels", "text"])
test=pd.read_csv("test.csv", header=0, names=["labels", "text"])
val=pd.read_csv("val.csv", header=0, names=["labels", "text"])
nlabels=train["labels"].nunique()

model_type = "bert"
model_name = "bert-base-cased"
use_cuda = True
train_batch_size = 100
eval_batch_size = 100
output_dir= "./our_bert"
use_early_stopping=1

args= {
"num_train_epochs" : 82,
"save_steps" : 5000,
"evaluate_during_training": True,
"overwrite_output_dir": True,
"save_best_model": True,
}


#model = ClassificationModel(model_type, "outputs/best_model", use_cuda=use_cuda, num_labels=nlabels, args=args)
model = ClassificationModel(model_type, "outputs/checkpoint-32000-epoch-45", use_cuda=use_cuda, num_labels=nlabels, args=args)

#model.train_model(train, args=args, eval_df=test, show_running_loss=True, accuracy=accuracy_score, p_r_f_s=partial(prfs, average=None))
result, model_outputs, wrong = model.eval_model(test, p_r_f_s=partial(prfs, average=None))
print(result)
#print(model_outputs)
#print(wrong)
