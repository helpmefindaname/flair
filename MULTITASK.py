import flair
from flair.data import Corpus
from flair.models.text_classification_model import TARSClassifier
from flair.trainers import ModelTrainer
from flair.datasets import TREC_50, CSVClassificationCorpus

def get_corpora():
    trec50_label_name_map = {'ENTY:sport': 'question about entity sport',
                             'ENTY:dismed': 'question about entity diseases medicine',
                             'LOC:city': 'question about location city',
                             'DESC:reason': 'question about description reasons',
                             'NUM:other': 'question about number other',
                             'LOC:state': 'question about location state',
                             'NUM:speed': 'question about number speed',
                             'NUM:ord': 'question about number order ranks',
                             'ENTY:event': 'question about entity event',
                             'ENTY:substance': 'question about entity element substance',
                             'NUM:perc': 'question about number percentage fractions',
                             'ENTY:product': 'question about entity product',
                             'ENTY:animal': 'question about entity animal',
                             'DESC:manner': 'question about description manner of action',
                             'ENTY:cremat': 'question about entity creative pieces inventions books',
                             'ENTY:color': 'question about entity color',
                             'ENTY:techmeth': 'question about entity technique method',
                             'NUM:dist': 'question about number distance measure',
                             'NUM:weight': 'question about number weight',
                             'LOC:mount': 'question about location mountains',
                             'HUM:title': 'question about person title',
                             'HUM:gr': 'question about person group organization of persons',
                             'HUM:desc': 'question about person description',
                             'ABBR:abb': 'question about abbreviation abbreviation',
                             'ENTY:currency': 'question about entity currency',
                             'DESC:def': 'question about description definition',
                             'NUM:code': 'question about number code',
                             'LOC:other': 'question about location other',
                             'ENTY:other': 'question about entity other',
                             'ENTY:body': 'question about entity body organ',
                             'ENTY:instru': 'question about entity musical instrument',
                             'ENTY:termeq': 'question about entity term equivalent',
                             'NUM:money': 'question about number money prices',
                             'NUM:temp': 'question about number temperature',
                             'LOC:country': 'question about location country',
                             'ABBR:exp': 'question about abbreviation expression',
                             'ENTY:symbol': 'question about entity symbol signs',
                             'ENTY:religion': 'question about entity religion',
                             'HUM:ind': 'question about person individual',
                             'ENTY:letter': 'question about entity letters characters',
                             'NUM:date': 'question about number date',
                             'ENTY:lang': 'question about entity language',
                             'ENTY:veh': 'question about entity vehicle',
                             'NUM:count': 'question about number count',
                             'ENTY:word': 'question about entity word special property',
                             'NUM:period': 'question about number period lasting time',
                             'ENTY:plant': 'question about entity plant',
                             'ENTY:food': 'question about entity food',
                             'NUM:volsize': 'question about number volume size',
                             'DESC:desc': 'question about description description'
                             }
    trec50: Corpus = TREC_50(label_name_map=trec50_label_name_map)

    column_name_map = {0: "label", 2: "text"}
    corpus_path = f"{flair.cache_root}/datasets/ag_news_csv"
    agnews_label_name_map = {'1': 'World',
                              '2': 'Sports',
                              '3': 'Business',
                              '4': 'Science Technology'
                              }
    agnews: Corpus = CSVClassificationCorpus(
        corpus_path,
        column_name_map,
        skip_header=False,
        delimiter=',',
        label_name_map=agnews_label_name_map
    )

    column_name_map = {0: "label", 2: "text"}
    corpus_path = f"{flair.cache_root}/datasets/dbpedia_csv"
    dbpedia_label_name_map = {'1': 'Company',
                      '2': 'Educational Institution',
                      '3': 'Artist',
                      '4': 'Athlete',
                      '5': 'Office Holder',
                      '6': 'Mean Of Transportation',
                      '7': 'Building',
                      '8': 'Natural Place',
                      '9': 'Village',
                      '10': 'Animal',
                      '11': 'Plant',
                      '12': 'Album',
                      '13': 'Film',
                      '14': 'Written Work'
                      }
    dbpedia: Corpus = CSVClassificationCorpus(
        corpus_path,
        column_name_map,
        skip_header=False,
        delimiter=',',
        label_name_map=dbpedia_label_name_map
    ).downsample(0.25)

    column_name_map = {0: "label", 2: "text"}
    corpus_path = f"{flair.cache_root}/datasets/amazon_review_full_csv"
    amazon_label_name_map = {'1': 'very negative product sentiment',
                      '2': 'negative product sentiment',
                      '3': 'neutral product sentiment',
                      '4': 'positive product sentiment',
                      '5': 'very positive product sentiment'
                      }
    amazon: Corpus = CSVClassificationCorpus(corpus_path,
                                             column_name_map,
                                             skip_header=False,
                                             delimiter=',',
                                             label_name_map=amazon_label_name_map
                                             ).downsample(0.05)

    column_name_map = {0: "label", 1: "text"}
    corpus_path = f"{flair.cache_root}/datasets/yelp_review_full_csv"
    yelp_label_name_map = {'1': 'very negative restaurant sentiment',
                      '2': 'negative restaurant sentiment',
                      '3': 'neutral restaurant sentiment',
                      '4': 'positive restaurant sentiment',
                      '5': 'very positive restaurant sentiment'
                      }
    yelp: Corpus = CSVClassificationCorpus(corpus_path,
                                           column_name_map,
                                           skip_header=False,
                                           delimiter=',',
                                           label_name_map=yelp_label_name_map
                                           ).downsample(0.25)

    return {"trec50":trec50,
            "agnews":agnews,
            "dbpedia":dbpedia,
            "amazon":amazon,
            "yelp":yelp}

def train_sequential_model(corpora, configurations):
    amazon_full = corpora.get("amazon")
    amazon_corpus = Corpus(train=amazon_full.train, dev=amazon_full.dev)
    tars = TARSClassifier(task_name='AMAZON', label_dictionary=amazon_corpus.make_label_dictionary(),
                          document_embeddings=configurations["model"])
    trainer = ModelTrainer(tars, amazon_corpus)
    trainer.train(base_path=f"{configurations['path']}/sequential_model/1_after_amazon",
                  learning_rate=0.02,
                  mini_batch_size=16,
                  max_epochs=10,
                  embeddings_storage_mode='none')

    yelp_full = corpora.get("yelp")
    yelp_corpus = Corpus(train=yelp_full.train, dev=yelp_full.dev)
    tars.add_and_switch_to_new_task("YELP", label_dictionary=yelp_corpus.make_label_dictionary())
    trainer = ModelTrainer(tars, yelp_corpus)
    trainer.train(base_path=f"{configurations['path']}/sequential_model/2_after_yelp",
                  learning_rate=0.02,
                  mini_batch_size=16,
                  max_epochs=10,
                  embeddings_storage_mode='none')

    dbpedia_full = corpora.get("dbpedia")
    dbpedia_corpus = Corpus(train=dbpedia_full.train, dev=dbpedia_full.dev)
    tars.add_and_switch_to_new_task("DBPEDIA", label_dictionary=dbpedia_corpus.make_label_dictionary())
    trainer = ModelTrainer(tars, dbpedia_corpus)
    trainer.train(base_path=f"{configurations['path']}/sequential_model/3_after_dbpedia",
                  learning_rate=0.02,
                  mini_batch_size=16,
                  max_epochs=10,
                  embeddings_storage_mode='none')

    agnews_full = corpora.get("agnews")
    agnews_corpus = Corpus(train=agnews_full.train, dev=agnews_full.dev)
    tars.add_and_switch_to_new_task("AGNEWS", label_dictionary=agnews_corpus.make_label_dictionary())
    trainer = ModelTrainer(tars, dbpedia_corpus)
    trainer.train(base_path=f"{configurations['path']}/sequential_model/4_after_agnews",
                  learning_rate=0.02,
                  mini_batch_size=16,
                  max_epochs=10,
                  embeddings_storage_mode='none')

    trec_full = corpora.get("trec50")
    trec_corpus = Corpus(train=trec_full.train, dev=trec_full.dev)
    tars.add_and_switch_to_new_task("TREC50", label_dictionary=trec_corpus.make_label_dictionary())
    trainer = ModelTrainer(tars, dbpedia_corpus)
    trainer.train(base_path=f"{configurations['path']}/sequential_model/5_after_trec",
                  learning_rate=0.02,
                  mini_batch_size=16,
                  max_epochs=10,
                  embeddings_storage_mode='none')

def train_multitask_model():
    mutlitask_model = {"trec": tars.add_and_switch_to_new_task()}
    MultitaskModel(tars, corpora)

if __name__ == "__main__":
    flair.device = "cuda:3"
    path_model_mapping = {
        "bert-base-uncased":
            {
                "path" : "2_bert_baseline",
                "model": "distilbert-base-uncased"
            },
        "bert-entailment-standard":
            {
                "path": "2_entailment_standard",
                "model": "distilbert_entailment/pretrained_mnli/best_model"
            },
        "bert-entailment-advanced":
            {
                "path": "2_entailment_advanced",
                "model": "distilbert_entailment/pretrained_mnli_rte_fever/best_model"
            }
    }
    corpora = get_corpora()
    for key, configurations in path_model_mapping.items():
        train_sequential_model(corpora, configurations)
        train_multitask_model()