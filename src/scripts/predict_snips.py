from allennlp.data.dataset_readers import CopyNetDatasetReader
from allennlp.predictors import Seq2SeqPredictor
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.models import load_archive


def main():
    dataset_reader = CopyNetDatasetReader(target_namespace='target_tokens',
                                          source_token_indexers={'tokens': SingleIdTokenIndexer(
                                              namespace='source_tokens'),
                                                                 'token_characters': TokenCharactersIndexer()})

    model_archive = load_archive(archive_file='checkpoints/model.tar.gz',
                                 cuda_device=-1,
                                 weights_file='checkpoints/model_state_epoch_28.th')

    model = model_archive.model
    model.eval()

    predictor = Seq2SeqPredictor(model=model,
                                 dataset_reader=dataset_reader)

    val_file = open('snips/val.tsv')
    for line in val_file:
        source, target = line.strip().split('\t')
        print('Gold Target: {}'.format(target.replace('OPEN', '(').replace('CLOSE', ')')))
        predicted_tokens = predictor.predict(target)['predicted_tokens'][0]
        print('Predictions: {}'.format(' '.join(predicted_tokens)).replace('OPEN', '(').replace('CLOSE', ')') + '\n')


if __name__ == '__main__':
    main()
