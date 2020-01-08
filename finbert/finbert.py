from __future__ import absolute_import, division, print_function

import random

import pandas as pd
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.nn import MSELoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm_notebook as tqdm
from tqdm import trange
from nltk.tokenize import sent_tokenize
from finbert.utils import *
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Config(object):
    """The configuration class for training."""

    def __init__(self,
                 data_dir,
                 bert_model,
                 model_dir,
                 max_seq_length=64,
                 train_batch_size=32,
                 eval_batch_size=32,
                 learning_rate=5e-5,
                 num_train_epochs=10.0,
                 warm_up_proportion=0.1,
                 no_cuda=False,
                 do_lower_case=True,
                 seed=42,
                 local_rank=-1,
                 gradient_accumulation_steps=1,
                 fp16=False,
                 output_mode='classification',
                 discriminate=True,
                 gradual_unfreeze=True,
                 encoder_no=12):
        """

        Parameters
        ----------
        data_dir: str
            Path for the training and evaluation datasets.
        bert_model: BertModel
            The BERT model to be used. For example: BertForSequenceClassification.from_pretrained(...)
        model_dir: str
            The path where the resulting model will be saved.
        max_seq_length: int
            The maximum length of the sequence to be used. Default value is 64.
        train_batch_size: int
            The batch size for the training. Default value is 32.
        eval_batch_size: int
            The batch size for the evaluation. Default value is 32.
        learning_rate: float
            The learning rate. Default value is 5e5.
        num_train_epochs: int
            Number of epochs to train. Default value is 4.
        warm_up_proportion: float
            During the training, the learning rate is linearly increased. This value determines when the learning rate
            reaches the intended learning rate. Default value is 0.1.
        no_cuda: bool
            Determines whether to use gpu. Default is False.
        do_lower_case: bool
            Determines whether to make all training and evaluation examples lower case. Default is True.
        seed: int
            Random seed. Defaults to 42.
        local_rank: int
            Used for number of gpu's that will be utilized. If set -1, no distributed training will be done. Default
            value is -1.
        gradient_accumulation_steps: int
            Number of gradient accumulations steps. Defaults to 1.
        fp16: bool
            Determines whether to use 16 bits for floats, instead of 32.
        output_mode: 'classification' or 'regression'
            Determines whether the task is classification or regression.
        discriminate: bool
            Determines whether to apply discriminative fine-tuning. 
        gradual_unfreeze: bool
            Determines whether to gradually unfreeze lower and lower layers as the training goes on.
        encoder_no: int
            Starting from which layer the model is going to be finetuned. If set 12, whole model is going to be
            fine-tuned. If set, for example, 6, only the last 6 layers will be fine-tuned.
        """
        self.data_dir = data_dir
        self.bert_model = bert_model
        self.model_dir = model_dir
        self.do_lower_case = do_lower_case
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.local_rank = local_rank
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.warm_up_proportion = warm_up_proportion
        self.no_cuda = no_cuda
        self.seed = seed
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.output_mode = output_mode
        self.fp16 = fp16   
        self.discriminate = discriminate        
        self.gradual_unfreeze = gradual_unfreeze
        self.encoder_no = encoder_no


class FinBert(object):
    """
    The main class for FinBERT.
    """

    def __init__(self,
                 config):
        self.config = config

    def prepare_model(self, label_list):
        """
        Sets some of the components of the model: Dataset processor, number of labels, usage of gpu and distributed
        training, gradient accumulation steps and tokenizer.

        Parameters
        ----------
        label_list: list
            The list of labels values in the dataset. For example: ['positive','negative','neutral']
        """

        self.processors = {
            "finsent": FinSentProcessor
        }

        self.num_labels_task = {
            'finsent': 2
        }

        if self.config.local_rank == -1 or self.config.no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not self.config.no_cuda else "cpu")
            self.n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(self.config.local_rank)
            self.device = torch.device("cuda", self.config.local_rank)
            self.n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            self.device, self.n_gpu, bool(self.config.local_rank != -1), self.config.fp16))

        if self.config.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                self.config.gradient_accumulation_steps))

        self.config.train_batch_size = self.config.train_batch_size // self.config.gradient_accumulation_steps

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.config.seed)

        if os.path.exists(self.config.model_dir) and os.listdir(self.config.model_dir):
            raise ValueError("Output directory ({}) already exists and is not empty.".format(self.config.model_dir))
        if not os.path.exists(self.config.model_dir):
            os.makedirs(self.config.model_dir)


        self.processor = self.processors['finsent']()
        self.num_labels = len(label_list)
        self.label_list = label_list

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=self.config.do_lower_case)

    def get_data(self, phase):
        """
        Gets the data for training or evaluation. It returns the data in the format that pytorch will process. In the
        data directory, there should be a .csv file with the name <phase>.csv
        Parameters
        ----------
        phase: str
            Name of the dataset that will be used in that phase. For example if there is a 'train.csv' in the data
            folder, it should be set to 'train'.

        Returns
        -------
        examples: list
            A list of InputExample's. Each InputExample is an object that includes the information for each example;
            text, id, label...
        """

        self.num_train_optimization_steps = None
        examples = None
        examples = self.processor.get_examples(self.config.data_dir, phase)
        self.num_train_optimization_steps = int(
            len(
                examples) / self.config.train_batch_size / self.config.gradient_accumulation_steps) * self.config.num_train_epochs
        
        if phase=='train':
            train = pd.read_csv(os.path.join(self.config.data_dir, 'train.csv'),sep='\t',index_col=False)
            weights = list()
            labels = self.label_list
    
            class_weights = [train.shape[0] / train[train.label == label].shape[0] for label in labels]
            self.class_weights = torch.tensor(class_weights)

        return examples

    def create_the_model(self):
        """
        Creates the model. Sets the model to be trained and the optimizer.
        """

        model = self.config.bert_model

        model.to(self.device)

        # Prepare optimizer
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        lr = self.config.learning_rate
        dft_rate = 1.2

        if self.config.discriminate:
            # apply the discriminative fine-tuning. discrimination rate is governed by dft_rate.

            encoder_params = []
            for i in range(12):
                encoder_decay = {
                    'params': [p for n, p in list(model.bert.encoder.layer[i].named_parameters()) if
                               not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01,
                    'lr': lr / (dft_rate ** (12 - i))}
                encoder_nodecay = {
                    'params': [p for n, p in list(model.bert.encoder.layer[i].named_parameters()) if
                               any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': lr / (dft_rate ** (12 - i))}
                encoder_params.append(encoder_decay)
                encoder_params.append(encoder_nodecay)

            optimizer_grouped_parameters = [
                {'params': [p for n, p in list(model.bert.embeddings.named_parameters()) if
                            not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01,
                 'lr': lr / (dft_rate ** 13)},
                {'params': [p for n, p in list(model.bert.embeddings.named_parameters()) if
                            any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0,
                 'lr': lr / (dft_rate ** 13)},
                {'params': [p for n, p in list(model.bert.pooler.named_parameters()) if
                            not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01,
                 'lr': lr},
                {'params': [p for n, p in list(model.bert.pooler.named_parameters()) if
                            any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0,
                 'lr': lr},
                {'params': [p for n, p in list(model.classifier.named_parameters()) if
                            not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01,
                 'lr': lr},
                {'params': [p for n, p in list(model.classifier.named_parameters()) if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0,
                 'lr': lr}]

            optimizer_grouped_parameters.extend(encoder_params)


        else:
            param_optimizer = list(model.named_parameters())

            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]


        schedule = "warmup_linear"



        self.optimizer = BertAdam(optimizer_grouped_parameters,
                                  lr=self.config.learning_rate,
                                  warmup=self.config.warm_up_proportion,
                                  t_total=self.num_train_optimization_steps,
                                  schedule=schedule)

        return model

    def get_loader(self, examples, phase):
        """
        Creates a data loader object for a dataset.

        Parameters
        ----------
        examples: list
            The list of InputExample's.
        phase: 'train' or 'eval'
            Determines whether to use random sampling or sequential sampling depending on the phase.

        Returns
        -------
        dataloader: DataLoader
            The data loader object.
        """

        features = convert_examples_to_features(examples, self.label_list,
                                                self.config.max_seq_length,
                                                self.tokenizer,
                                                self.config.output_mode)

        # Log the necessasry information
        logger.info("***** Loading data *****")
        logger.info("  Num examples = %d", len(examples))
        logger.info("  Batch size = %d", self.config.train_batch_size)
        logger.info("  Num steps = %d", self.num_train_optimization_steps)

        # Load the data, make it into TensorDataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        if self.config.output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        elif self.config.output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

        try:
            all_agree_ids = torch.tensor([f.agree for f in features], dtype=torch.long)
        except:
            all_agree_ids = torch.tensor([0.0 for f in features], dtype=torch.long)

        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_agree_ids)

        # Distributed, if necessary
        if phase == 'train':
            my_sampler = RandomSampler(data)
        elif phase == 'eval':
            my_sampler = SequentialSampler(data)

        dataloader = DataLoader(data, sampler=my_sampler, batch_size=self.config.train_batch_size)
        return dataloader

    def train(self, train_examples, model):
        """
        Trains the model.
        Parameters
        ----------
        examples: list
            Contains the data as a list of InputExample's
        model: BertModel
            The Bert model to be trained.
        weights: list
            Contains class weights.
        Returns
        -------
        model: BertModel
            The trained model.
        """
        
        
        validation_examples = self.get_data('validation')

        global_step = 0



        self.validation_losses = []

        # Training
        train_dataloader = self.get_loader(train_examples, 'train')

        model.train()

        step_number = len(train_dataloader)


        i = 0
        for _ in trange(int(self.config.num_train_epochs), desc="Epoch"):

            model.train()

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):

                if (self.config.gradual_unfreeze and i == 0):
                    for param in model.bert.parameters():
                        param.requires_grad = False

                if (step % (step_number // 3)) == 0:
                    i += 1

                if (self.config.gradual_unfreeze and i > 1 and i < self.config.encoder_no):

                    for k in range(i - 1):

                        try:
                            for param in model.bert.encoder.layer[self.config.encoder_no - 1 - k].parameters():
                                param.requires_grad = True
                        except:
                            pass

                if (self.config.gradual_unfreeze and i > self.config.encoder_no + 1):
                    for param in model.bert.embeddings.parameters():
                        param.requires_grad = True

                batch = tuple(t.to(self.device) for t in batch)

                input_ids, input_mask, segment_ids, label_ids, agree_ids = batch

                logits = model(input_ids, segment_ids, input_mask)
                weights = self.class_weights.to(self.device)

                if self.config.output_mode == "classification":
                    loss_fct = CrossEntropyLoss(weight = weights)
                    loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                elif self.config.output_mode == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.fp16:
                        lr_this_step = self.config.learning_rate * warmup_linear(
                            global_step / self.num_train_optimization_steps, self.config.warmup_proportion)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

            # Validation

            validation_loader = self.get_loader(validation_examples, phase='eval')
            model.eval()

            valid_loss, valid_accuracy = 0, 0
            nb_valid_steps, nb_valid_examples = 0, 0

            for input_ids, input_mask, segment_ids, label_ids, agree_ids in tqdm(validation_loader, desc="Validating"):
                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)
                segment_ids = segment_ids.to(self.device)
                label_ids = label_ids.to(self.device)
                agree_ids = agree_ids.to(self.device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask)

                    if self.config.output_mode == "classification":
                        loss_fct = CrossEntropyLoss(weight=weights)
                        tmp_valid_loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                    elif self.config.output_mode == "regression":
                        loss_fct = MSELoss()
                        tmp_valid_loss = loss_fct(logits.view(-1), label_ids.view(-1))

                    valid_loss += tmp_valid_loss.mean().item()

                    nb_valid_steps += 1

            valid_loss = valid_loss / nb_valid_steps

            self.validation_losses.append(valid_loss)
            print("Validation losses: {}".format(self.validation_losses))

            if valid_loss == min(self.validation_losses):

                try:
                    os.remove(self.config.model_dir / ('temporary' + str(best_model)))
                except:
                    print('No best model found')
                torch.save({'epoch': str(i), 'state_dict': model.state_dict()},
                           self.config.model_dir / ('temporary' + str(i)))
                best_model = i

        # Save a trained model and the associated configuration
        checkpoint = torch.load(self.config.model_dir / ('temporary' + str(best_model)))
        model.load_state_dict(checkpoint['state_dict'])
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(self.config.model_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(self.config.model_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())
        os.remove(self.config.model_dir / ('temporary' + str(best_model)))
        return model

    def evaluate(self, model, examples):
        """
        Evaluate the model.
        Parameters
        ----------
        model: BertModel
            The model to be evaluated.
        examples: list
            Evaluation data as a list of InputExample's/

        Returns
        -------
        evaluation_df: pd.DataFrame
            A dataframe that includes for each example predicted probability and labels.
        """

        eval_loader = self.get_loader(examples, phase='eval')

        logger.info("***** Running evaluation ***** ")
        logger.info("  Num examples = %d", len(examples))
        logger.info("  Batch size = %d", self.config.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        predictions = []
        labels = []
        agree_levels = []
        text_ids = []

        for input_ids, input_mask, segment_ids, label_ids, agree_ids in tqdm(eval_loader, desc="Testing"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            label_ids = label_ids.to(self.device)
            agree_ids = agree_ids.to(self.device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask)

                if self.config.output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    tmp_eval_loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                elif self.config.output_mode == "regression":
                    loss_fct = MSELoss()
                    tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

                np_logits = logits.cpu().numpy()

                if self.config.output_mode == 'classification':
                    prediction = np.array(np_logits)
                elif self.config.output_mode == "regression":
                    prediction = np.array(np_logits)

                for agree_id in agree_ids:
                    agree_levels.append(agree_id.item())

                for label_id in label_ids:
                    labels.append(label_id.item())

                for pred in prediction:
                    predictions.append(pred)

                text_ids.append(input_ids)

                # tmp_eval_loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                # tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)

                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

            # logits = logits.detach().cpu().numpy()
            # label_ids = label_ids.to('cpu').numpy()
            # tmp_eval_accuracy = accuracy(logits, label_ids)

            # eval_loss += tmp_eval_loss.mean().item()
            # eval_accuracy += tmp_eval_accuracy

        evaluation_df = pd.DataFrame({'predictions': predictions, 'labels': labels, "agree_levels": agree_levels})

        return evaluation_df


def predict(text, model, write_to_csv=False, path=None):
    """
    Predict sentiments of sentences in a given text. The function first tokenizes sentences, make predictions and write
    results.

    Parameters
    ----------
    text: string
        text to be analyzed
    model: BertForSequenceClassification
        path to the classifier model
    write_to_csv (optional): bool
    path (optional): string
        path to write the string
    """
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    sentences = sent_tokenize(text)

    label_list = ['positive', 'negative', 'neutral']
    label_dict = {0: 'positive', 1: 'negative', 2: 'neutral'}
    result = pd.DataFrame(columns=['sentence','logit','prediction','sentiment_score'])
    for batch in chunks(sentences, 5):

        examples = [InputExample(str(i), sentence) for i, sentence in enumerate(batch)]

        features = convert_examples_to_features(examples, label_list, 64, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        with torch.no_grad():
            logits = model(all_input_ids, all_segment_ids, all_input_mask)
            logits = softmax(np.array(logits))
            sentiment_score = pd.Series(logits[:,0] - logits[:,1])
            predictions = np.squeeze(np.argmax(logits, axis=1))

            batch_result = {'sentence': batch,
                            'logit': list(logits),
                            'prediction': predictions,
                            'sentiment_score':sentiment_score}
            
            batch_result = pd.DataFrame(batch_result)
            result = pd.concat([result,batch_result], ignore_index=True)

    result['prediction'] = result.prediction.apply(lambda x: label_dict[x])
    if write_to_csv:
        result.to_csv(path,sep=',', index=False)

    return result
