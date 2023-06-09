
import numpy as np
import cv2
import os
import re
from tqdm import tqdm
import pandas as pd
import sys
import csv
import pickle
import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer, AutoModel
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import emoji
import unicodedata

from flask import Flask, request ,jsonify

from flask_cors import CORS



# # Create the BertClassfier class

# # Create a function to tokenize a set of texts


# Specify `MAX_LEN`
MAX_LEN =  280
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

class EmptyModel(nn.Module):
    def __init__(self):
        super(EmptyModel, self).__init__()
        # No layers or parameters to initialize

    def forward(self, x):
        # No computation to perform
        return x

# Create an instance of the empty model


def initialize_model(epochs=4, version="mini"):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(freeze_bert=False, version=version)
    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(params=list(bert_classifier.parameters()),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler

class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False, version="mini"):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in = 256 if version == "mini" else 768
        H, D_out = 50, 2

        # Instantiate BERT model
        self.bert = AutoModel.from_pretrained("asafaya/bert-mini-arabic") if version == "mini" else AutoModel.from_pretrained("asafaya/bert-base-arabic")
        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits






def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
  

    # Normalize unicode encoding
    text = unicodedata.normalize('NFC', text)
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    #Remove URLs
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '<URL>', text)


    return text

def remove_emojis(sent):
    text =  emoji.demojize(sent)
    text= re.sub(r'(:[!_\-\w]+:)', '', text)
    return text

def text_preprocessing_no_emojis(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
  
    # Remove emojis
    text = remove_emojis(text)

    return text_preprocessing(text)

def preprocessing_for_bert(data, version="mini", text_preprocessing_fn = text_preprocessing ):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []
    tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-mini-arabic") if version == "mini" else AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")

    # For every sentence...
    for i,sent in enumerate(data):
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing_fn(sent),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,                  # Max length to truncate/pad
            padding='max_length',        # Pad sentence to max length
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True,     # Return attention mask
            truncation = True 
            )
        
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks







def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs

# Helper function to get the prediction of a single tweet's sentiment (can be used for random tweet testing)
def predict_tweet_sentiment(tweet):
    df = pd.DataFrame([tweet])
    df = df.rename(columns = {0:"tweet"})
    print(df.tweet.values)
    test_inputs, test_masks = preprocessing_for_bert(df.tweet.values)

    # Create the DataLoader for our test set
    test_dataset = TensorDataset(test_inputs, test_masks)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)
    # Compute predicted probabilities on the test set
    probs = bert_predict(bert_classifier, test_dataloader)
    print(probs)
    # Get predictions from the probabilities
    threshold = 0.5
    preds = np.where(probs[:, 1] > threshold, "positive", "negative")
    

#     print("no-negative tweets ratio ", preds.sum()/len(preds))
    return preds, probs
import nltk
from nltk.corpus import stopwords
def text_preprocessing_1(s):
    """
    - Lowercase the sentence
    - Change "'t" to "not"
    - Remove "@name"
    - Isolate and remove punctuations except "?"
    - Remove other special characters
    - Remove stop words except "not" and "can"
    - Remove trailing whitespace
    """
    

    nltk.download('stopwords')

    # Now you can use the 'stopwords' variable

    stop_words = stopwords.words('english')
    s = s.lower()
    # Change 't to 'not'
    s = re.sub(r"\'t", " not", s)
    # Remove @name
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    # Isolate and remove punctuations except '?'
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Remove stopwords except 'not' and 'can'
    s = " ".join([word for word in s.split()
                  if word not in stopwords.words('english')
                  or word in ['not', 'can']])
    # Remove trailing whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    
    return s
def text_preprocessing_2(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    text = re.sub(r'https', '', text)
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

from langdetect import detect
import re
import csv
from getpass import getpass
from time import sleep
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

def predict_tweet_sentiment(model,tweet):
    df = pd.DataFrame([tweet])
    df = df.rename(columns = {0:"tweet"})
    print(df.tweet.values)
    test_inputs, test_masks = preprocessing_for_bert(df.tweet.values)

    # Create the DataLoader for our test set
    test_dataset = TensorDataset(test_inputs, test_masks)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)
    # Compute predicted probabilities on the test set
    probs = bert_predict(model, test_dataloader)
    print(probs)
    max_index = np.argmax(probs)
    
    return  probs[max_index],max_index

def get_tweet_data(card):
    """Extract data from tweet card"""
    username = card.find_element(By.XPATH,'.//span').text
    try:
        handle = card.find_element(By.XPATH,'.//span[contains(text(), "@")]').text
    except NoSuchElementException:
        return
    
    try:
        postdate = card.find_element(By.XPATH,'.//time').get_attribute('datetime')
    except NoSuchElementException:
        return
    
    comment = card.find_element(By.XPATH,'/html/body/div[1]/div/div/div[2]/main/div/div/div/div/div/div[3]/div/section/div/div/div[1]/div/div/article/div/div/div[2]/div[2]/div[2]/div').text
    text = comment


    
    tweet = text 
    return tweet  


def scrapping(hachtag,num):
    hach = '#'+hachtag
    # create instance of web driver

    options = Options()
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    driver = webdriver.Chrome(options=options)

    # navigate to login screen
    driver.get('https://twitter.com/i/flow/login?input_flow_data=%7B%22requested_variant%22%3A%22eyJsYW5nIjoiZnIifQ%3D%3D%22%7D')
    driver.maximize_window()
    sleep(5)

    # find search input and search for term
    search_input_mail = driver.find_element(By.XPATH,'//input[@name="text"]')
    search_input_mail.send_keys('mn.nedjah@gmail.com')
    search_input_mail.send_keys(Keys.RETURN)
    sleep(1)
    #password
    try:
        search_input_password = driver.find_element(By.XPATH,'//input[@name="password"]')
        search_input_password.send_keys('mahmoud2030')
        search_input_password.send_keys(Keys.RETURN)
        sleep(1)
    except:
        # find search input and search for term
        search_input_us = driver.find_element(By.XPATH,'//input[@name="text"]')
        search_input_us.send_keys('MnNazih')
        search_input_us.send_keys(Keys.RETURN)
        sleep(1)
        search_input_password = driver.find_element(By.XPATH,'//input[@name="password"]')
        search_input_password.send_keys('mahmoud2030')
        search_input_password.send_keys(Keys.RETURN)
        sleep(1)

    # find search input and search for term
    sleep(5)
    search_input = driver.find_element(By.XPATH,'//input[@aria-label="Search query"]')
    search_input.send_keys(hach)
    search_input.send_keys(Keys.RETURN)
    sleep(5)
    # Locate the element by link text and click on it
    element = driver.find_element(By.LINK_TEXT, 'Latest')
    element.click()
    scroll_attempt = 0

    
    data = []
    tweet_ids = set()
    last_position = driver.execute_script("return window.pageYOffset;")
    scroll_attempt = 0
    scrolling = True
    sleep(5)

    while scrolling:
        page_cards = driver.find_elements(By.XPATH, '//article[@data-testid="tweet"]')
        for card in page_cards[-15:]:
            
            tweet = get_tweet_data(card)  # Replace with appropriate code to extract tweet data
            tweet_id = ''.join(tweet)
            data.append(tweet)

        driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')

        scroll_attempt += 1

        if scroll_attempt >= num:
            curr_position = driver.execute_script("return window.pageYOffset;")
            scrolling = False
        else:
            sleep(2)  # Attempt another scroll

    # Close the web driver
    #driver.close()
    print(data)
    unique_list = list(set(data))
    df = pd.DataFrame()
    df['text']= unique_list
    df['processed'] = df['text'].apply( text_preprocessing_2)
    df['processed'] = df['processed'].apply( text_preprocessing_1)
    i=[]
    for text in df['text']:
        if (detect(text) == 'ar'):
            config,scores = arabic(text)
            i.append(np.argmax(scores))
        else:
            config,scores = english(text)
            i.append(np.argmax(scores))
    df['sentiment']=i
    df.to_csv('./output.csv', index=False)
    neg = (df['sentiment'] == 1).sum()
    pos = (df['sentiment'] == 0).sum()
    net = (df['sentiment'] == 2).sum()

    return pos,neg ,net
 
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

def english(txt):

    # Preprocess text (username and link placeholders)
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    
    
    #model.save_pretrained(MODEL)
    #text = preprocesss(txt)
    encoded_input = tokenizer(txt, return_tensors='pt')
    output = english_model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    return config,scores

def arabic(txt):

    # Preprocess text (username and link placeholders)
    MODEL = f"ALANZI/imamu_arabic_sentimentAnalysis"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    # PT
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    #model.save_pretrained(MODEL)
    #   text = preprocess(text)
    encoded_input = tokenizer(txt, return_tensors='pt')
    output = arabic_model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    return config,scores


# Paths to load the model


#Load the Model
print("im here")
#bert_classifier, optimizer, scheduler = initialize_model(epochs=2, version="base")

print("im here2")
#bert_classifier = BertClassifier(freeze_bert=True, version="base")


# filename = 'BERT_base_no_emojis_ArSAS (1).sav'
# f = open(filename, 'rb')
# bert_classifier = pickle.load(f)

filename = 'BERT_base_no_emojis_ArSAS22.pth'


# Loading the model
#bert_classifier = torch.load(filename,map_location=torch.device('cpu'))

english_model = torch.load('bert_classifier_model_finall.pth',map_location=torch.device('cpu'))
arabic_model = torch.load('bert_classifier_model_finall_ar.pth',map_location=torch.device('cpu'))

# here = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(here)

# with open('bert_classifier.pkl', 'rb') as file:
#     bert_classifier = pickle.load(file)




print("model loaded")
# Step 4: Close the file
#f.close()





# # Load centers for ab channel quantization used for rebalancing.

app = Flask(__name__)
CORS(app)




@app.route('/process', methods=['POST'])
def process_text():
    if request.method == 'POST':
        # Get the text data from the request
        text = request.json.get('text')
        
        if text is not None:
            # Process the text (perform any desired operations)
            processed_text = text.upper()
            #senti= predict_tweet_sentiment(text)
            #sentimento , proba = senti
            
            sentiment=[]
            percnt = []
            if (detect(text) == 'ar'):
                config,scores = arabic(text)
                ranking = np.argsort(scores)
                ranking = ranking[::-1]
                for i in range(scores.shape[0]):
                    l = config.id2label[ranking[i]]
                    s = scores[ranking[i]]
                    sentiment.append(l)
                    percnt.append(np.round(float(s), 4))
            
            else:   
                config,scores = english(text)
                ranking = np.argsort(scores)
                ranking = ranking[::-1]
                for i in range(scores.shape[0]):
                    l = config.id2label[ranking[i]]
                    s = scores[ranking[i]]
                    sentiment.append(l)
                    percnt.append(np.round(float(s), 4))

            #print("this is sentimento", str(s))
            #print("this is proba:", str(proba_eng))
            # Return the processed text as a response
            # return str(predict_tweet_sentiment(text))
            response = {'result': sentiment,
                        'proba':percnt
                        }

            return jsonify(response)
        else:
            print("no text provideddddd")
            return 'No text provided.'
    

@app.route('/process_hashtag', methods=['POST'])

def process_hashtag():
    if request.method == 'POST':
        # Get the text data from the request
        hashtag = request.json.get('hashtag')
        number = request.json.get('number')
        num_neg,num_net,num_pos = scrapping(hashtag,number)
        print('hi',num_neg,num_net,num_pos)
        
        response = {'pos':str(num_pos) ,
                    'neg':str(num_neg),
                    'net':str(num_net)
                        }
        return jsonify(response)
       




if __name__ == '__main__':
    app.run()

