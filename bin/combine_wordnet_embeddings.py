
import torch
import numpy as np
import h5py

from allennlp.models.archival import load_archive
from kb.common import JsonFile
import simplejson as json
import kb.kg_embedding
from nltk.corpus import wordnet as wn



# includes @@PADDING@@, @@UNKNOWN@@, @@MASK@@, @@NULL@@
NUM_EMBEDDINGS = 117663

def generate_wordnet_synset_vocab(entity_file, vocab_file):
    vocab = ['@@UNKNOWN@@']
    
    with JsonFile(entity_file, 'r') as fin:
        for node in fin:
            if node['type'] == 'synset':
                vocab.append(node['id'])

    vocab.append('@@MASK@@')
    vocab.append('@@NULL@@')

    with open(vocab_file, 'w') as fout:
        fout.write('\n'.join(vocab))


def extract_tucker_embeddings(tucker_archive, vocab_file, tucker_hdf5):
    archive = load_archive(tucker_archive)

    with open(vocab_file, 'r') as fin:
        vocab_list = fin.read().strip().split('\n')

    # get embeddings
    embed = archive.model.kg_tuple_predictor.entities.weight.detach().numpy()
    # print(f'embed.shape {embed.shape}')
    # print(embed[0])
    out_embeddings = np.zeros((NUM_EMBEDDINGS, embed.shape[1]))
    
    # print(f'out_embeddings.shape {out_embeddings.shape}')

    vocab = archive.model.vocab


    for k, entity in enumerate(vocab_list):
        embed_id = vocab.get_token_index(entity, 'entity')
        # print(f'enumer(vocab_list) {k} vocab.get_token_index {embed_id}')
        if entity in ('@@MASK@@', '@@NULL@@'):
            # these aren't in the tucker vocab -> random init
            out_embeddings[k + 1, :] = np.random.randn(1, embed.shape[1]) * 0.004
        elif entity != '@@UNKNOWN@@':
            assert embed_id != 1
            # k = 0 is @@UNKNOWN@@, and want it at index 1 in output
            out_embeddings[k + 1, :] = embed[embed_id, :]

    print(out_embeddings[1357])
    # write out to file
    with h5py.File(tucker_hdf5, 'w') as fout:
        ds = fout.create_dataset('tucker', data=out_embeddings)


def debias_tucker_embeddings(tucker_archive, tucker_hdf5, vocab_file):
    #Get tucker embeddings numpy array
    with h5py.File(tucker_hdf5, 'r') as fin:
        tucker = fin['tucker'][...]

    #Get list of words in vocabulary
    with open(vocab_file, 'r') as fin:
        vocab_list = fin.read().strip().split('\n')

    #Extract gender directional vectors
    archive = load_archive(tucker_archive)
    vocab = archive.model.vocab
    gender_dir_vecs = []
    for info in ["n.01", "n.02", "a.01", "s.02", "s.03"]:
        # id1 = vocab.get_token_index('female.'+info, 'entity')
        # id2 = vocab.get_token_index('male.'+info, 'entity')
        k1 = [k for k,entity in enumerate(vocab_list) if entity.startswith('female.'+info)]
        k2 = [k for k,entity in enumerate(vocab_list) if entity.startswith('male.'+info)]
        print(f'k1 {k1} k2 {k2}')
        gender_dir_vecs.append(tucker[k1[0]+1, :]-tucker[k2[0]+1,:])
    lambdas = [0.2]*5   #Parameters which decide the amount of debiasing


    #Debias tucker embeddings of job titles and traits
    for filename in ["job_titles.txt", "negative_traits", "positive_traits"]:
        f = open("bin/debiasing_words/refined_"+filename, "r")
        for line in f.readlines():
            target_word = line.strip().lower().replace(" - ","_").replace("-","_").replace(" ","_")
            idxs = [k for k,entity in enumerate(vocab_list) if entity.startswith(target_word+".")]
            print(f'len(idxs) {len(idxs)}')
            for k in idxs: #For each entity corresponding to each word
                # id = vocab.get_token_index(entity, 'entity')
                # print(f'entity {entity}, id {id}')
                # if k<0 or k>=NUM_EMBEDDINGS:
                #     continue
                for i in range(len(gender_dir_vecs)):  #Debias the embedding
                    gdv = gender_dir_vecs[i]
                    lam = lambdas[i]
                    tucker[k+1, :] = tucker[k+1, :] - gdv * lam * np.dot(tucker[k+1, :], gdv) / np.linalg.norm(gdv)

    #Write to the new embeddings file.
    with h5py.File('tucker_embeddings/debiased/e100_ldot2.hdf5', 'w') as fout:
        ds = fout.create_dataset('tucker', data=tucker)


def create_refined_word_lists():
    vocab_file = "wordnet_synsets_mask_null_vocab.txt"
    with open(vocab_file, 'r') as fin:
        vocab_list = fin.read().strip().split('\n')
    for filename in ["job_titles.txt", "negative_traits", "positive_traits"]:
        print(f'{filename}')
        num_debiased_word = 0
        num_notfound_words = 0
        not_found_words = []
        f = open("bin/debiasing_words/"+filename, "r")
        new_f = open("bin/debiasing_words/refined_"+filename, "a")
        for line in f.readlines():
            target_word = line.strip().lower().replace(" - ","_").replace("-","_").replace(" ","_")
            entities = [w for w in vocab_list if w.startswith(target_word+".")]
            if len(entities)>0:
                num_debiased_word += 1
                new_f.write(target_word+"\n")
            else:
                num_notfound_words += 1
                not_found_words.append(target_word)
                # print(f'Not found: {target_word}')
        print(f'Found: {num_debiased_word}')
        print(f'Not found: {num_notfound_words}')
        print(not_found_words)

    # print(f'debiased {num_debiased_word} words, {num_debiased_embeddings} embeddings')
    # print(f'Num not found: {num_notfound_words}')


def get_gensen_synset_definitions(entity_file, vocab_file, gensen_file):
    from gensen import GenSen, GenSenSingle

    gensen_1 = GenSenSingle(
        model_folder='./data/models',
        filename_prefix='nli_large_bothskip',
        pretrained_emb='./data/embedding/glove.840B.300d.h5'
    )
    gensen_1.eval()

    definitions = {}
    with open(entity_file, 'r') as fin:
        for line in fin:
            node = json.loads(line)
            if node['type'] == 'synset':
                definitions[node['id']] = node['definition']

    with open(vocab_file, 'r') as fin:
        vocab_list = fin.read().strip().split('\n')

    # get the descriptions
    sentences = [''] * NUM_EMBEDDINGS
    for k, entity in enumerate(vocab_list):
        definition = definitions.get(entity)
        if definition is None:
            assert entity in ('@@UNKNOWN@@', '@@MASK@@', '@@NULL@@')
        else:
            sentences[k + 1] = definition

    embeddings = np.zeros((NUM_EMBEDDINGS, 2048), dtype=np.float32)
    for k in range(0, NUM_EMBEDDINGS, 32):
        sents = sentences[k:(k+32)]
        reps_h, reps_h_t = gensen_1.get_representation(
            sents, pool='last', return_numpy=True, tokenize=True
        )
        embeddings[k:(k+32), :] = reps_h_t
        print(k)

    with h5py.File(gensen_file, 'w') as fout:
        ds = fout.create_dataset('gensen', data=embeddings)


def combine_tucker_gensen(tucker_hdf5, gensen_hdf5, all_file):
    with h5py.File(tucker_hdf5, 'r') as fin:
        tucker = fin['tucker'][...]

    with h5py.File(gensen_hdf5, 'r') as fin:
        gensen = fin['gensen'][...]

    all_embeds = np.concatenate([tucker, gensen], axis=1)
    all_e = all_embeds.astype(np.float32)

    with h5py.File(all_file, 'w') as fout:
        ds = fout.create_dataset('tucker_gensen', data=all_e)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_wordnet_synset_vocab', default=False, action="store_true")
    parser.add_argument('--entity_file', type=str)
    parser.add_argument('--vocab_file', type=str)

    parser.add_argument('--generate_gensen_embeddings', default=False, action="store_true")
    parser.add_argument('--gensen_file', type=str)

    parser.add_argument('--extract_tucker', default=False, action="store_true")
    parser.add_argument('--tucker_archive_file', type=str)
    parser.add_argument('--tucker_hdf5_file', type=str)

    parser.add_argument('--debias_tucker', default=False, action="store_true")

    parser.add_argument('--combine_tucker_gensen', default=False, action="store_true")
    parser.add_argument('--all_embeddings_file', type=str)

    args = parser.parse_args()


    if args.generate_wordnet_synset_vocab:
        generate_wordnet_synset_vocab(args.entity_file, args.vocab_file)
    elif args.generate_gensen_embeddings:
        get_gensen_synset_definitions(args.entity_file, args.vocab_file, args.gensen_file)
    elif args.extract_tucker:
        extract_tucker_embeddings(args.tucker_archive_file, args.vocab_file, args.tucker_hdf5_file)
    elif args.combine_tucker_gensen:
        combine_tucker_gensen(args.tucker_hdf5_file, args.gensen_file, args.all_embeddings_file)
    elif args.debias_tucker:
        debias_tucker_embeddings(args.tucker_archive_file, args.tucker_hdf5_file, args.vocab_file)


