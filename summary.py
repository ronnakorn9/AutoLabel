### code example from https://huggingface.co/ml6team/keyphrase-extraction-kbir-inspec

import copy
import numpy as np
from sklearn import metrics

from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline
)
from transformers.pipelines import AggregationStrategy
import torch
from torch.utils.data import TensorDataset, DataLoader

from classifier import multiclass_classifier, classifier_fit

class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model="ml6team/keyphrase-extraction-kbir-inspec", *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, all_outputs):
        results = super().postprocess(
            all_outputs=all_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        sentence = all_outputs[0]["sentence"]
        return np.unique([select_word(sentence, start_pos=result["start"], end_pos=result["end"]-1) for result in results])
        
# class entity_remover():
#     def __init__(self):
#         self.classifier = pipeline("ner", model="stevhliu/my_awesome_wnut_model")

#     def remove_entity(self, text):
#         return self.classifier(text)

class entity_remover():
    def __init__(self):

        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER-uncased")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER-uncased")

        self.classifier = pipeline("ner", model=model, tokenizer=tokenizer)

    def get_entity(self, text):
        return self.classifier(text)
    

def label_propagate(embedding:np.ndarray, current_label:np.ndarray):
    """
    apply actual label (1 ... k) to noise label (-1)
    by training a classification model on avaliable label and infer the missing label
    """
    assert(len(embedding) == len(embedding))
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # perpare classifier model
    num_class = len(np.unique(current_label[current_label != -1]))
    input_size = embedding.shape[1]
    model = multiclass_classifier(input_size=input_size, num_classes=num_class)

    # prepare the data
    condition = current_label != -1
    X_train = embedding[condition]
    X_test = embedding[~condition]
    y_train = current_label[condition]
    y_test = current_label[~condition]

    # convert numpy input to PyTorch Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    model = classifier_fit(model, X_train_tensor, y_tensor, device=device)

    # infer missing label
    model.eval()
    with torch.no_grad():
        model.eval()  # Set model to evaluation mode
        X_test_tensor = X_test_tensor.to(device)
        outputs = model(X_test_tensor)
        _, predictions = torch.max(outputs, 1)
    predicted = predictions.cpu().numpy()

    new_label = copy.deepcopy(current_label)
    new_label[~condition] = predicted
    return embedding, new_label


def print_cluster_metrics(predicted_cluster, true_cluster):
    """
    print clustering score
    """
    h_score = metrics.homogeneity_score(true_cluster, predicted_cluster)
    c_score = metrics.completeness_score(true_cluster, predicted_cluster)
    nmf = metrics.normalized_mutual_info_score(true_cluster, predicted_cluster)
    amf = metrics.adjusted_mutual_info_score(true_cluster, predicted_cluster)
    ars = metrics.adjusted_rand_score(true_cluster, predicted_cluster)
    print(f"homogeneity_score: {h_score:.2f}")
    print(f"completeness_score: {c_score:.2f}")
    print(f"normalized_mutual_info_score: {nmf:.2f}")
    print(f"adjusted_mutual_info_score: {amf:.2f}")
    print(f"adjusted_rand_score: {ars:.2f}")

def generate_candidate_intent(corpus, corpus_embedding, current_label:np.ndarray, sample_size=5):
    """
    gnerating intent candidate for zero-shot learning
    """
    
    corpus = np.array(corpus)
    candidate_labels = []

    unique_intent = np.unique(current_label)
    # for entity detection
    remover = entity_remover()

    # for keyword extraction
    model = "ml6team/keyphrase-extraction-kbir-openkp"
    extractor = KeyphraseExtractionPipeline(model=model)

    # extracting intent candidate for each cluster
    for intent in unique_intent:
        word_set = set()
        if intent == -1:
            continue

        cluster_corpus = corpus[current_label == intent]
        # sample a few data for each intent
        sampled_data = np.random.choice(cluster_corpus, size=sample_size, replace=False)
        closest_point, closest_index = closest_point_to_centroid(corpus_embedding[current_label == intent])
        sampled_data = np.append(sampled_data, [cluster_corpus[closest_index]], axis=0)
    
        # extract keyword from text        
        keyphrases = extract_keyword(extractor, sampled_data.tolist())

        # remove entity name
        keyphrases = remove_entity_from_keyphase(remover, sampled_data, keyphrases)
        for word_list in keyphrases:
            word_set.update(word_list)

        # remove duplicate ex. ['ground transportation', 'transportation'] => ['transportation']
        word_set = remove_superset_words(list(word_set))
        
        ### merge the keyphase
        new_intent = ""
        for word in word_set:
            if new_intent == "":
                new_intent = word
            else:
                new_intent = new_intent + "+" + word
        
        candidate_labels.append(new_intent)

    # remove duplicate again
    candidate_labels = remove_superset_words(candidate_labels)
    return list(set(candidate_labels))

def infer_label(corpus, candidate_labels):
    classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
    
    output = classifier(corpus, candidate_labels)
    new_label = [dict_item["labels"][0] for dict_item in output]
    return new_label

def label_propagate_transformer(corpus, corpus_embeddings:np.ndarray, current_label:np.ndarray):
    
    candidate_intent = generate_candidate_intent(corpus, corpus_embeddings, current_label)
    # candidate_intent = list(set(candidate_intent))

    predicted_label = infer_label(corpus, candidate_intent)

    return corpus, predicted_label

def extract_keyword(extractor, corpus):
    """
    corpus: list of text
    return 
    """
    # model = "ml6team/keyphrase-extraction-kbir-openkp"
    # extractor = KeyphraseExtractionPipeline(model=model)
    keyphrases = extractor(corpus)

    # return list of numpy.ndarray, each of which is array of string
    return keyphrases


def closest_point_to_centroid(data):
    # Compute centroid
    centroid = np.mean(data, axis=0)
    
    # Calculate distances to centroid for each data point
    distances = np.linalg.norm(data - centroid, axis=1)
    
    # Find index of closest point
    closest_index = np.argmin(distances)
    
    # Return closest data point
    closest_point = data[closest_index]
    
    return closest_point, closest_index

####### Token classification #######
def remove_entity_from_keyphase(entity_remover:entity_remover, text_list, keyphrases):
    
    for i, text in enumerate(text_list):
        word_dict = entity_remover.get_entity(text)

        ban_word_list = []
        for item in word_dict:
            ban_word_list.append(item['word'])

        keyphrases[i] = remove_elements_with_banned_strings(keyphrases[i], ban_word_list)

    return keyphrases
    

def remove_elements_with_banned_strings(word_list, ban_string_list):
    ban_string_set = set(ban_string_list)
    
    # Filter elements based on whether they contain banned strings
    filtered_list = [elem for elem in word_list if not any(ban_string in elem for ban_string in ban_string_set)]
    
    return filtered_list

def remove_superset_words(word_list):
    """
    Remove words from a list if they are supersets of smaller words in the same list.

    Args:
    - word_list: List of words

    Returns:
    - filtered_list: List with superset words removed
    """
    # Sort the word list by length in descending order
    sorted_words = sorted(word_list, key=len, reverse=True)
    
    # Initialize an empty set to store the words to be removed
    words_to_remove = set()
    
    # Iterate over each word in the sorted list
    for word in sorted_words:
        # Check if any smaller word is a subset of the current word
        for smaller_word in sorted_words:
            if len(smaller_word) < len(word) and smaller_word in word:
                words_to_remove.add(word)
                break  # Break the inner loop if a smaller word is found
    
    # Create a filtered list without the superset words
    filtered_list = [word for word in word_list if word not in words_to_remove]
    
    return filtered_list

def select_word(sentence, start_pos, end_pos):
    """
    Select the whole word from a sentence given its start and end positions.

    Args:
    - sentence: Input sentence as a string
    - start_pos: Start position of the word
    - end_pos: End position of the word

    Returns:
    - selected_word: The selected word
    """
    # Find the start of the word
    while start_pos > 0 and sentence[start_pos - 1].isalnum():
        start_pos -= 1
    
    # Find the end of the word
    while end_pos < len(sentence) - 1 and sentence[end_pos + 1].isalnum():
        end_pos += 1
    
    # Extract the word from the sentence
    selected_word = sentence[start_pos:end_pos + 1]
    
    return selected_word