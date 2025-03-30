"shortjokes.csv" : The short joke dataset used for this project contains more than 200,000 short jokes.


"30shortjokes.csv" : Truncated or filled to a short joke dataset of 30 characters by "cut30.py".


"Sjoks.csv" : A dataset of 200,000 data was selected from "30shortjokes.csv" to train the model and verify the output results of the model.


"train_data.csv" : The training dataset.


"test_data.csv" : The testing dataset.


"vocab.txt" : The vocabulary. This file saved the 50,000 words that appear most frequently in the data set, which are used to generate jokes when using the model.


"validation_data.csv" : The validation dataset.


"Statistic.py" : Used to calculate the number of data in each length interval in the original data set.


"cut30.py" : Used to uniformly truncate or fill jokes in the original data set to a length of 30 characters.


"Select.py" : It is used to randomly select 200,000 pieces of data from "cut30.py" in order to shorten the training time of the model and facilitate the memory of the model.


"Clean Split CSV.py" : Used to split training sets, test sets and verification sets from "Sjos.csv".


"Training.py" : For the architecture and training of the model, two model files "best_model.pkl" and "final_model.pkl" will be generated. The former is the model with the best training effect and the latter is the model after the last round of training iteration.


"USE.py" : Used to load and use the trained model, where "best_model.pkl" is used.


"best_model.pkl" : This is the model file that works best.


"final_model.pkl" : This is the model file at the end of all training iterations.