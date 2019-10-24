import json

SNIPS_INTENTS = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic', 'RateBook', 'SearchCreativeWork',
                 'SearchScreeningEvent']


def main():
    train_examples = []
    val_examples = []
    for intent in SNIPS_INTENTS:
        with open('snips/train_{}_full.json'.format(intent), encoding='utf-8', errors='replace') as inf:
            int_dict = json.load(inf)
            for example in int_dict[intent]:
                source = ' '.join([' '.join(item['text'].lower().split()) for item in example['data']])
                target = intent + ' ' + ' '.join([item['entity'] + 'OPEN ' + ' '.join(item['text'].lower().split()) + ' CLOSE'
                                                  + item['entity'] for item in example['data'] if 'entity' in item])
                train_examples.append((source, target))

        with open('snips/validate_{}.json'.format(intent), encoding='utf-8', errors='replace') as inf:
            int_dict = json.load(inf)
            for example in int_dict[intent]:
                source = ' '.join([' '.join(item['text'].lower().split()) for item in example['data']])
                target = intent + ' ' + ' '.join([item['entity'] + 'OPEN ' + ' '.join(item['text'].lower().split()) + ' CLOSE'
                                                  + item['entity'] for item in example['data'] if 'entity' in item])
                val_examples.append((source, target))

    with open('snips/train.tsv', 'w') as out_f:
        for item in train_examples:
            out_str = item[0] + '\t' + item[1] + '\n'
            out_f.write(out_str)

    with open('snips/val.tsv', 'w') as out_f:
        for item in val_examples:
            out_str = item[0] + '\t' + item[1] + '\n'
            out_f.write(out_str)


if __name__ == '__main__':
    main()
