# kaggle_quora (active competition)

<p> In this Kaggle competition, the goal is to compile a model to identify if a pair of questioins is asking the same thing or not. Quora provided 400K+ question pairs for 
the training set, and the final test data set has 2,345,796 question pairs (that's alot of data!). While many Kagglers have used techniques such as Xgboost and feature extraction such as TF-IDF, ratio of matching words, and weighted word2vec, recurrent neural network is used here to explore its potential in solving this problem.
Log loss is used to evaluate the performance of the model.</p>

## Recurrent Neural Net (LSTM) ##
<p> For background on recurrent neural net and its differnet derivations, Google Brain research scientist Christopher Colah provides excellent <a href="http://colah.github.io/posts/2015-08-Understanding-LSTMs/">
explanation</a> complete with clear visualization. In short, recurrent network works well with temporal-dependent data where a particular data point is dependent on previous data point(s). Examples of such data set are stock prices,  sentence and semantics, etc. Recurrent network has chain like structure composed of repeating units, and each unit (after time 0) accepts the data point AND the output of the previous unit as the input. 
LSTM neural net is a recurrent network with individual unit that contains a <i>memory cell</i> which runs through the entire chain. This
memory cell works with other elements in the unit, such as forget-gate and input-gate, to decide what information/feature to keep, thus learning the semantic relevance of each word in the case of sentence input. This architecture allows LSTM to capture information few units prior and the
memory cell avoids the vanishing-gradient problem common to recurrent neural net.</p>
<p align='center'><img src = 'LSTM3-chain.png', width=60%, height=60%><br> Fig 1. Design of LSTM neural network(source: Colah). For the input of this data set, X_t corresponds to the word at t position of the sentence, and X_t+1 is the next word and so on.</p>

<p> In this code, the model input would be a pair of questions and it outputs a prediction where 1 is duplicate. Each question is embedded using the Glove pretrained word vector, and each embedded vector is fed to 
a LSTM network. Then the representation output from the LSTM layer is combined to calculate the distance 
(the sum of the squared difference between the two representation vectors), and that goes through two dense layers with the final dense layer 
being the sigmoid function. This model architecture is similar to Siamese network, except there's the final sigmoid function to predict a binary outcome, and 
the model is trained by minimizing log loss. In the tradition Siamese network, the output of the model is simply the distance between the two 
representation vectors and contrastive loss is used to train the model.

