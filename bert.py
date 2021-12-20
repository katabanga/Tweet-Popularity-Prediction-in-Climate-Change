import pandas as pd
from transformers import BertModel
from transformers import BertTokenizer
import torch
import torch.nn as nn
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
import nltk
import jsonlines
from nltk.corpus import words
from nltk.corpus import wordnet 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import random
import time
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


class DataType:
    string_type = ['tweet', 'description', 'user_location']
    float_type = ['num_tag', 'num_emoji', 'follower_count', 'friends_count', 'retweet_count', 'num_at']
    boolean_value = ['has_num', 'has_mark', 'has_link']

class Input2Model:
    def __init__(self, df):
        self.df = df
        self.feature_input = []
        self.tweet_max_length = 128
        self.bio_max_length = 128
        self.location_max_length = 16

    def clean_tweet(self):
        nltk.download('words')
        nltk.download('wordnet')
        nltk.download('stopwords')
        nltk.download('punkt')

        word_net = set(wordnet.words())
        word_ = set(words.words())
        total_words = set.union(word_net,word_)
        stop_words = set(stopwords.words('english'))


        for index, row in tqdm(self.df.iterrows()):
            text = row[4]
            text_tokens = word_tokenize(text)
            tokens_without_sw = [word for word in text_tokens if not word.lower() in stopwords.words()]
            filtered_sentence = (" ").join(tokens_without_sw)
            
        # for text in tqdm(self.tweet):
            num_words = len(text.split())
            newtext = " ".join(w for w in nltk.wordpunct_tokenize(text) \
                    if w.lower() in total_words or not w.isalpha())
            num_english = len(newtext.split())
            if num_english/num_words < 0.75:
                df.drop(index, inplace=True)

    def model_input(self):
        for each in self.df.text.values:
            tweet_features = Tweet2Features(each)
            num_tag, has_num, has_mark, num_at, num_emoji = tweet_features.feature_style()
            model_feature_input = {
                'num_emoji': num_emoji, # float
                'num_tag': num_tag,  # float
                'num_at': num_at, # float
                'has_num': has_num, # bool
                'has_mark': has_mark, # bool
            }
            self.feature_input.append(model_feature_input)
        meta_info_df = self.df[['text', 'description', 'user_location', 'follower_count', 'friends_count', 'retweet_count', 'has_link', 'num_of_likes']]
        feature_input_df = pd.DataFrame(self.feature_input)
        meta_info_df = meta_info_df.reset_index(drop=True)
        model_input_df = pd.concat([meta_info_df, feature_input_df], axis=1)
        train_df, val_df = train_test_split(model_input_df, test_size=0.1, random_state=2020)
        return train_df, val_df

    def preprocess_for_BERT(self, given_df):
        tweet_input_ids = []
        tweet_attention_masks = []
        user_bio_ids = []
        user_bio_attention_masks = []
        user_location_ids = []
        user_location_attention_masks = []
        tweet = given_df.text.values
        user_bio = given_df.description.values
        user_location = given_df.user_location.values
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        for i in range(len(tweet)):
            encoded_tweet = tokenizer.encode_plus(
                text=tweet[i],    # Preprocess sentence
                add_special_tokens=True,          # Add `[CLS]` and `[SEP]`
                max_length=self.tweet_max_length, # Max length to truncate/pad
                pad_to_max_length=True,           # Pad sentence to max length
                return_attention_mask=True        # Return attention mask
            )
            tweet_input_ids.append(encoded_tweet.get('input_ids'))
            tweet_attention_masks.append(encoded_tweet.get('attention_mask'))

            encoded_user_bio = tokenizer.encode_plus(
                text=user_bio[i],    # Preprocess sentence
                add_special_tokens=True,          # Add `[CLS]` and `[SEP]`
                max_length=self.bio_max_length, # Max length to truncate/pad
                pad_to_max_length=True,           # Pad sentence to max length
                return_attention_mask=True        # Return attention mask
            )
            user_bio_ids.append(encoded_user_bio.get('input_ids'))
            user_bio_attention_masks.append(encoded_user_bio.get('attention_mask'))

            encoded_user_location = tokenizer.encode_plus(
                text=user_location[i],    # Preprocess sentence
                add_special_tokens=True,          # Add `[CLS]` and `[SEP]`
                max_length=self.location_max_length, # Max length to truncate/pad
                pad_to_max_length=True,           # Pad sentence to max length
                return_attention_mask=True        # Return attention mask
            )
            user_location_ids.append(encoded_user_location.get('input_ids'))
            user_location_attention_masks.append(encoded_user_location.get('attention_mask'))

        return tweet_input_ids, tweet_attention_masks, user_bio_ids, user_bio_attention_masks, user_location_ids, user_location_attention_masks

class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, tweet_input_ids, tweet_attention_masks, user_bio_ids, user_bio_attention_masks, user_location_ids, user_location_attention_masks, df):
        self.tweet_input_ids = tweet_input_ids
        self.tweet_attention_masks = tweet_attention_masks
        self.user_bio_ids = user_bio_ids
        self.user_bio_attention_masks = user_bio_attention_masks
        self.user_location_ids = user_location_ids
        self.user_location_attention_masks = user_location_attention_masks
        self.num_emoji = df.num_emoji.values
        self.follower_count = df.follower_count.values
        self.has_mark = df.has_mark.values 
        self.has_num = df.has_num.values
        self.retweet_count = df.retweet_count.values
        self.num_at = df.num_at.values
        self.has_link = df.has_link.values
        self.num_tag = df.num_tag.values
        self.friends_count = df.friends_count.values 
        self.num_of_likes = df.num_of_likes.values
    
    def __len__(self):
        # print(len(self.tweet_input_ids))
        return len(self.tweet_input_ids)
    
    def __getitem__(self, index):
        # x = []
        tweet_ids = np.asarray(self.tweet_input_ids[index]).reshape(-1,1)
        tweet_attn = np.asarray(self.tweet_attention_masks[index]).reshape(-1,1)
        tweet = (tweet_ids, tweet_attn)

        user_bio_ids = np.asarray(self.user_bio_ids[index]).reshape(-1,1)
        user_bio_attn = np.asarray(self.user_bio_attention_masks[index]).reshape(-1,1)
        user_bio = (user_bio_ids, user_bio_attn)

        user_location_ids = np.asarray(self.user_location_ids[index]).reshape(-1,1)
        user_location_attn = np.asarray(self.user_location_attention_masks[index]).reshape(-1,1)
        user_location = (user_location_ids, user_location_attn)
      
        y = self.num_of_likes[index]

        return {'tweet': tweet, 'description':user_bio, 'user_location':user_location, 'num_emoji': self.num_emoji[index], 'follower_count': self.follower_count[index], 'has_mark': self.has_mark[index], 'has_num': self.has_num[index], 'retweet_count': self.retweet_count[index],
                'num_at': self.num_at[index], 'friends_count': self.friends_count[index], 'has_link': self.has_link[index], 'num_tag': self.num_tag[index], 'num_of_likes': y}

class BertClassifier(nn.Module):
    def __init__(self, num_string, num_float, num_bool, output_dimension, freeze_bert=False):
        super(BertClassifier, self).__init__()
        float_hidden, float_out = 8, 16
        bool_out_size = 8
        self.string2emb = BertModel.from_pretrained('bert-base-uncased')
        self.float2emb = nn.Sequential(
            nn.Linear(1, float_hidden),
            nn.ReLU(),
            nn.Linear(float_hidden, float_out),
        )
        self.bool2emb = nn.Linear(num_bool, bool_out_size)
        output_size = 768 * num_string + float_out * num_float
        if num_bool != 0:
          output_size += 8
        h1, h2, h3 = 512, 256, 64

        self.classifier = nn.Sequential(
            nn.Linear(output_size, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(h3, output_dimension)
        )
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        # layers

    def forward(self, batch, selected_keys):
        '''
        each batch is
        {(input_ids, attention_mask)
            'tweet': [[input_ids1, input_ids2 ...], [attention_mask1, attention_mask2, ...]], # string
            'abstract': [abstract1, abstract2, abstract3, ...], # string
            'text_feature1': [True, False, True],
            'text_feature2': [True, False, True],
            'text_feature3': [True, False, True],
            'publication_year': float, # every int should be returned to float
        }
        '''
        bool_concat = []
        first = True
        datatype = DataType()
        output = torch.empty((2,2))
        # print(batch)
        for key, value in batch.items():
            if key in selected_keys:
                if key in datatype.float_type:
                    value = value.float().to(device)
                    float_emb = self.float2emb(value.reshape(-1,1))
                    if first:
                        output = float_emb
                        first = False
                    else:
                        output = torch.cat((output, float_emb),1)
                elif key in datatype.string_type:
                    string_emb = self.string2emb(input_ids=value[0][:,:,0].to(device), attention_mask=value[1][:,:,0].to(device))
                    last_hidden_state_cls = string_emb[0][:, 0, :]
                    if first:
                        output = last_hidden_state_cls
                        first = False
                    else:
                        output = torch.cat((output, last_hidden_state_cls),1)
                else:
                    bool_concat.append(value.float().to(device))
    
        if bool_concat:
            num_bool = len(bool_concat)
            batch_size = len(bool_concat[0])
            bool_array = torch.empty([num_bool,batch_size])
            for i in range(num_bool):
                  bool_array[i] = torch.from_numpy(np.array(bool_concat[i].cpu()))
            bool_array = bool_array.to(device)
            bool_emb = self.bool2emb(torch.transpose(bool_array,0,1))
            output = output = torch.cat((output, bool_emb),1)
        labels = batch['num_of_likes'].to(device)
        logits = self.classifier(output)
        return logits, labels

class Tweet2Features:
    def __init__(self, tweet):
        self.tweet = tweet
        # self.tokens = word_tokenize(title)

    def feature_style(self):
        pattern_tag = '#\s\w+'
        pattern_number = '\d'
        pattern_mark = '[\?\!]'
        pattern_at = '@\s[a-zA-Z0-9]+'
        pattern_emoji = re.compile('[#*0-9]️⃣|[©®‼⁉™ℹ↔-↙↩↪⌚⌛⌨⏏⏩-⏳⏸-⏺Ⓜ▪▫▶◀◻-◾☀-☄☎☑☔☕☘]|☝[🏻-🏿]?|[☠☢☣☦☪☮☯☸-☺♀♂♈-♓♟♠♣♥♦♨♻♾♿⚒-⚗⚙⚛⚜⚠⚡⚪⚫⚰⚱⚽⚾⛄⛅⛈⛎⛏⛑⛓⛔⛩⛪⛰-⛵⛷⛸]|⛹(?:️‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|[⛺⛽✂✅✈✉]|[✊-✍][🏻-🏿]?|[✏✒✔✖✝✡✨✳✴❄❇❌❎❓-❕❗❣❤➕-➗➡➰➿⤴⤵⬅-⬇⬛⬜⭐⭕〰〽㊗㊙🀄🃏🅰🅱🅾🅿🆎🆑-🆚]|🇦[🇨-🇬🇮🇱🇲🇴🇶-🇺🇼🇽🇿]|🇧[🇦🇧🇩-🇯🇱-🇴🇶-🇹🇻🇼🇾🇿]|🇨[🇦🇨🇩🇫-🇮🇰-🇵🇷🇺-🇿]|🇩[🇪🇬🇯🇰🇲🇴🇿]|🇪[🇦🇨🇪🇬🇭🇷-🇺]|🇫[🇮-🇰🇲🇴🇷]|🇬[🇦🇧🇩-🇮🇱-🇳🇵-🇺🇼🇾]|🇭[🇰🇲🇳🇷🇹🇺]|🇮[🇨-🇪🇱-🇴🇶-🇹]|🇯[🇪🇲🇴🇵]|🇰[🇪🇬-🇮🇲🇳🇵🇷🇼🇾🇿]|🇱[🇦-🇨🇮🇰🇷-🇻🇾]|🇲[🇦🇨-🇭🇰-🇿]|🇳[🇦🇨🇪-🇬🇮🇱🇴🇵🇷🇺🇿]|🇴🇲|🇵[🇦🇪-🇭🇰-🇳🇷-🇹🇼🇾]|🇶🇦|🇷[🇪🇴🇸🇺🇼]|🇸[🇦-🇪🇬-🇴🇷-🇹🇻🇽-🇿]|🇹[🇦🇨🇩🇫-🇭🇯-🇴🇷🇹🇻🇼🇿]|🇺[🇦🇬🇲🇳🇸🇾🇿]|🇻[🇦🇨🇪🇬🇮🇳🇺]|🇼[🇫🇸]|🇽🇰|🇾[🇪🇹]|🇿[🇦🇲🇼]|[🈁🈂🈚🈯🈲-🈺🉐🉑🌀-🌡🌤-🎄]|🎅[🏻-🏿]?|[🎆-🎓🎖🎗🎙-🎛🎞-🏁]|🏂[🏻-🏿]?|[🏃🏄](?:‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|[🏅🏆]|🏇[🏻-🏿]?|[🏈🏉]|🏊(?:‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|[🏋🏌](?:️‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|[🏍-🏰]|🏳(?:️‍🌈)?|🏴(?:‍☠️|󠁧󠁢(?:󠁥󠁮󠁧|󠁳󠁣󠁴|󠁷󠁬󠁳)󠁿)?|[🏵🏷-👀]|👁(?:️‍🗨️)?|[👂👃][🏻-🏿]?|[👄👅]|[👆-👐][🏻-🏿]?|[👑-👥]|[👦👧][🏻-🏿]?|👨(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?👨|[🌾🍳🎓🎤🎨🏫🏭]|👦(?:‍👦)?|👧(?:‍[👦👧])?|[👨👩]‍(?:👦(?:‍👦)?|👧(?:‍[👦👧])?)|[💻💼🔧🔬🚀🚒🦰-🦳])|[🏻-🏿](?:‍(?:[⚕⚖✈]️|[🌾🍳🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?)?|👩(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?[👨👩]|[🌾🍳🎓🎤🎨🏫🏭]|👦(?:‍👦)?|👧(?:‍[👦👧])?|👩‍(?:👦(?:‍👦)?|👧(?:‍[👦👧])?)|[💻💼🔧🔬🚀🚒🦰-🦳])|[🏻-🏿](?:‍(?:[⚕⚖✈]️|[🌾🍳🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?)?|[👪-👭]|👮(?:‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|👯(?:‍[♀♂]️)?|👰[🏻-🏿]?|👱(?:‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|👲[🏻-🏿]?|👳(?:‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|[👴-👶][🏻-🏿]?|👷(?:‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|👸[🏻-🏿]?|[👹-👻]|👼[🏻-🏿]?|[👽-💀]|[💁💂](?:‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|💃[🏻-🏿]?|💄|💅[🏻-🏿]?|[💆💇](?:‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|[💈-💩]|💪[🏻-🏿]?|[💫-📽📿-🔽🕉-🕎🕐-🕧🕯🕰🕳]|🕴[🏻-🏿]?|🕵(?:️‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|[🕶-🕹]|🕺[🏻-🏿]?|[🖇🖊-🖍]|[🖐🖕🖖][🏻-🏿]?|[🖤🖥🖨🖱🖲🖼🗂-🗄🗑-🗓🗜-🗞🗡🗣🗨🗯🗳🗺-🙄]|[🙅-🙇](?:‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|[🙈-🙊]|🙋(?:‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|🙌[🏻-🏿]?|[🙍🙎](?:‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|🙏[🏻-🏿]?|[🚀-🚢]|🚣(?:‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|[🚤-🚳]|[🚴-🚶](?:‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|[🚷-🚿]|🛀[🏻-🏿]?|[🛁-🛅🛋]|🛌[🏻-🏿]?|[🛍-🛒🛠-🛥🛩🛫🛬🛰🛳-🛹🤐-🤗]|[🤘-🤜][🏻-🏿]?|🤝|[🤞🤟][🏻-🏿]?|[🤠-🤥]|🤦(?:‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|[🤧-🤯]|[🤰-🤶][🏻-🏿]?|🤷(?:‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|[🤸🤹](?:‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|🤺|🤼(?:‍[♀♂]️)?|[🤽🤾](?:‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|[🥀-🥅🥇-🥰🥳-🥶🥺🥼-🦢🦰-🦴]|[🦵🦶][🏻-🏿]?|🦷|[🦸🦹](?:‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|[🧀-🧂🧐]|[🧑-🧕][🏻-🏿]?|🧖(?:‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|[🧗-🧝](?:‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|[🧞🧟](?:‍[♀♂]️)?|[🧠-🧿]')

        num_tag = len(re.findall(pattern_tag, self.tweet))
        has_num = re.search(pattern_number, self.tweet)!=None
        has_mark = re.search(pattern_mark, self.tweet)!=None
        num_at = len(re.findall(pattern_at, self.tweet))
        num_emoji = len(re.findall(pattern_emoji, self.tweet))

        return num_tag, has_num, has_mark, num_at, num_emoji


    def feature_topic(self, pre_calculated_features):
        # pre_calculated_features is an instance of FeaturesByNLPModels()
        return {
            'title2subarea': pre_calculated_features.title2subarea[self.title],
        }

    def feature_novelty(self, common_word_set):
        # common_word_set is what you pre-calculate on all paper title+abstract (tokenized version) of words larger than 100 occurrences
        # import nltk
        # new_text = nltk.word_tokenize(text)
        # We should avoid double tokenization: new_new_text = nltk.word_tokenize(new_text)

        low_freq_words = {i for i in self.tokens if i not in common_word_set}
        num_low_freq_words = len(low_freq_words)
        return {
            'num_low_freq_words': num_low_freq_words,
        }  # a number that shows how many low-frequency words are in the title.

def initialize_model(num_string, num_float, num_bool, output_dimension, epochs=4):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(num_string, num_float, num_bool, output_dimension, freeze_bert=False)

    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=1e-5,    # Default learning rate
                      eps=1e-4,    # Default epsilon value
                      weight_decay=1e-3
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, train_dataloader, selected_keys, val_dataloader=None, epochs=4, evaluation=False):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            # b_input_ids, b_attn_mask, a_input_ids, a_attn_mask, train_meta, b_labels = tuple(t.to(device) for t in batch)
            # Zero out any previously calculated gradients
            model.zero_grad()
            # Perform a forward pass. This will return logits.
            logits, labels = model(batch, selected_keys)

            # Compute loss and accumulate the loss values
            # labels = batch['num_of_likes'].to(device)
            loss = loss_fn(logits, labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader, selected_keys)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
    
    print("Training complete!")


def evaluate(model, val_dataloader, selected_keys):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        # b_input_ids, b_attn_mask, a_input_ids, a_attn_mask, val_meta, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits, labels = model(batch, selected_keys)

        # labels = batch['num_of_likes'].to(device)
        # Compute loss
        loss = loss_fn(logits, labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

if __name__ == "__main__":

  if torch.cuda.is_available():       
    device = torch.device("cuda") 
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

  else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

  df = pd.read_json('cleaned_tweet.jsonl', lines=True)
  df.drop_duplicates(subset ="text", keep = False, inplace = True)
  # loss_fn = nn.CrossEntropyLoss()

  df = df[df['num_of_likes'] < 4000]
  df = df[df['num_of_likes'] > 100]
  # df['num_of_likes'].describe()
  df['num_of_likes'] = (df['num_of_likes'] >= 356).astype(int)

  input2model = Input2Model(df)

  train_df, val_df = input2model.model_input()
  train_df.reset_index()
  train_tweet_input_ids, train_tweet_attention_masks, train_user_bio_ids, train_user_bio_attention_masks, train_user_location_ids, train_user_location_attention_masks = input2model.preprocess_for_BERT(train_df)
  val_tweet_input_ids, val_tweet_attention_masks, val_user_bio_ids, val_user_bio_attention_masks, val_user_location_ids, val_user_location_attention_masks = input2model.preprocess_for_BERT(val_df)

  batch_size = 32
  train_data = TweetDataset(train_tweet_input_ids, train_tweet_attention_masks, train_user_bio_ids, train_user_bio_attention_masks, train_user_location_ids, train_user_location_attention_masks, train_df)

  train_sampler = RandomSampler(train_data)
  train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, drop_last=True)

  val_data = TweetDataset(val_tweet_input_ids, val_tweet_attention_masks, val_user_bio_ids, val_user_bio_attention_masks, val_user_location_ids, val_user_location_attention_masks, val_df)

  val_sampler = RandomSampler(val_data)
  val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size, drop_last=True)

  selected_keys = ['tweet']
  print(selected_keys)
  print("lr = 1e-5, eps = 1e-4, weightdecay = 1e-3, numwarmup = 0, h3 = 64, batch = 32")

  loss_fn = nn.CrossEntropyLoss()
  set_seed(42)    # Set seed for reproducibility
  bert_classifier, optimizer, scheduler = initialize_model(num_string = 1, num_float = 0, num_bool = 0, output_dimension = 2)
  train(bert_classifier, train_dataloader, selected_keys, val_dataloader, epochs=6, evaluation=True)

