# AVASAG
data_selection.py splits the dataset into train and test, saves them in json files called test_data.json and train_data.json and we can use them for experiments!
This way the same train and test datasets will be used and the results of experiments are comparable!

In the file save_best_performing_model.py, I read the train data and use it for training the model from scratch! 
I ran this file two times with changing the epoch number and you have the results. The model weights in this cases are on the cluster and sorry I don't have access to them anymore!
I saved both best and last epoch model weights in a .pt file to compare their performance.
I also saved the training logs and outputs!

The file inference.py would use the saved model from address model_address and the test data to evaluate the model performance! and saves the results in a file containing the "predictions" in its name.

The whole model architecture considers all the inflection parameters, but I was also interested to see how it performs with only gloss information. That's why there is code and results called only_gloss.

The results can be found on this link:
https://dfkide-my.sharepoint.com/:u:/r/personal/semo06_dfki_de/Documents/Attachments/avasag%20results%20from%20server%201.zip?csf=1&web=1&e=lVpvEJ
