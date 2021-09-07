import json


def flatten_data(file):
    # Opening JSON file & returns JSON object as a dictionary
    f = open(file, encoding='utf-8')
    data = json.load(f)

    # Iterating through the json list 
    entry_list = list()

    for row in data['data']:
        title = row['title']

        for paragraph in row['paragraphs']:
            context = paragraph['context']

            for qa in paragraph['qas']:
                entry = {}

                qa_id = qa['id']
                question = qa['question']
                answers = qa['answers']

                entry['id'] = qa_id
                entry['title'] = title.strip()
                entry['context'] = context.strip()
                entry['question'] = question.strip()

                answer_starts = [answer['answer_start'] for answer in answers]
                answer_texts = [answer['text'].strip() for answer in answers]
                entry['answers'] = {}
                entry['answers']['answer_start'] = answer_starts
                entry['answers']['text'] = answer_texts

                entry_list.append(entry)

    reverse_entry_list = entry_list[::-1]

    # for entries with same id, keep only last one (corrected texts by the group Deep Learning Brasil)
    unique_ids_list = list()
    unique_entry_list = list()
    for entry in reverse_entry_list:
        qa_id = entry['id']
        if qa_id not in unique_ids_list:
            unique_ids_list.append(qa_id)
            unique_entry_list.append(entry)

    f.close()

    file_name = file[:file.rfind('/')] + '/flat_' + file[file.rfind('/')+1:]
    with open(file_name, 'w') as json_file:
        json.dump({'data': unique_entry_list}, json_file)

    return file_name


def prepare_train_features(examples, tokenizer, max_length, stride, padding_right):
    tokenized_examples = tokenizer(
        examples['question' if padding_right else 'context'],
        examples['context' if padding_right else 'question'],
        max_length=max_length, stride=stride, padding='max_length',
        return_overflowing_tokens=True, return_offsets_mapping=True,
        truncation='only_second' if padding_right else 'only_first')

    sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
    offset_mapping = tokenized_examples['offset_mapping']

    tokenized_examples['id'] = []
    tokenized_examples['start_positions'] = []
    tokenized_examples['end_positions'] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples['input_ids'][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples['id'].append(examples['id'][sample_index])
        answers = examples['answers'][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers['answer_start']) == 0:
            tokenized_examples['start_positions'].append(cls_index)
            tokenized_examples['end_positions'].append(cls_index)
            continue

        start_char = answers['answer_start'][0]
        end_char = start_char + len(answers['text'][0])

        context_token = 1 if padding_right else 0
        token_start_index = sequence_ids.index(context_token)
        token_end_index = len(sequence_ids) - sequence_ids[::-1].index(context_token) - 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if start_char < offsets[token_start_index][0] or end_char > offsets[token_end_index][1]:
            tokenized_examples['start_positions'].append(cls_index)
            tokenized_examples['end_positions'].append(cls_index)
            continue

        # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
        # Note: we could go after the last offset if the answer is the last word (edge case).
        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        tokenized_examples['start_positions'].append(token_start_index - 1)
        while offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        tokenized_examples['end_positions'].append(token_end_index + 1)

    return tokenized_examples