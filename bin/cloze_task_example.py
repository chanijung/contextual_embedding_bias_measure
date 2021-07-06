from kb.include_all import ModelArchiveFromParams
from kb.knowbert_utils import KnowBertBatchifier
from allennlp.common import Params

import torch

if __name__ == '__main__':
    # a pretrained model, e.g. for Wordnet+Wikipedia
    archive_file = 'https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wiki_wordnet_model.tar.gz'
    params = Params({"archive_file": archive_file})

    # load model and batcher
    model = ModelArchiveFromParams.from_params(params=params)
    batcher = KnowBertBatchifier(archive_file, masking_strategy='full_mask')

    # sentences = ["Paris is located in [MASK].", "La Mauricie National Park is located in [MASK]."]
    # sentences = ["[MASK] is good at programming"]
    sentences = ["[MASK] is good at [MASK]"]

    mask_id = batcher.tokenizer_and_candidate_generator.bert_tokenizer.vocab['[MASK]']

    # batcher takes raw untokenized sentences
    # and yields batches of tensors needed to run KnowBert
    for batch in batcher.iter_batches(sentences):
        model_output = model(**batch)
        token_mask = batch['tokens']['tokens'] == mask_id

        # print(f'pooled output size {model_output["pooled_output"].size()}')
        print(f'pooled output[0]\n{model_output["pooled_output"][0,:]}')
        # print(f'cont emb size {model_output["contextual_embeddings"].size()}')
        print(f'cont emb[0]\n{model_output["contextual_embeddings"][0,0,:]}')

        # (batch_size, timesteps, vocab_size)
        prediction_scores, _ = model.pretraining_heads(
                model_output['contextual_embeddings'], model_output['pooled_output']
        )

        # print(f'pred score size {prediction_scores.size()}')
        print(f'pred score[0]\n{prediction_scores[0,0,:]}')

        mask_token_probabilities = prediction_scores.masked_select(token_mask.unsqueeze(-1)).view(-1, prediction_scores.shape[-1])  # (num_masked_tokens, vocab_size)
        print(f'mask_token_prob size {mask_token_probabilities.size()}')

        predicted_token_ids = mask_token_probabilities.argmax(dim=-1)

        predicted_tokens = [batcher.tokenizer_and_candidate_generator.bert_tokenizer.ids_to_tokens[int(i)]
            for i in predicted_token_ids]

        # print(predicted_tokens)

