import argparse
import torch
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import save_file
from torch_kmeans import KMeans


def tokenize(prompts: list, tokenizer):
    token_list = []
    for prompt in prompts:
        token_list.append(tokenizer(prompt).input_ids[1:-1])
    return token_list


def get_vectors(token_list, text_encoder):
    vector_list = []
    embs = text_encoder.get_input_embeddings().weight.data
    if token_list:
        if len(token_list) > 1:
            for token_set in token_list:
                temp_vectors = []
                for token in token_set:
                    temp_vectors.append(embs[token:token + 1])
                vector_list.append(torch.cat(temp_vectors).mean(dim=0, keepdim=True))
        else:
            for token in token_list[0]:
                vector_list.append(embs[token:token + 1])
    return vector_list


def negate(pos_vectors, neg_vectors):
    vector_list = []
    for vector in pos_vectors:
        for neg_vector in neg_vectors:
            vector -= (vector * neg_vector).sum() / (neg_vector * neg_vector).sum() * neg_vector
        vector_list.append(vector)
    return vector_list


def cluster(vector_list, model):
    temp_list = model(torch.cat(vector_list).unsqueeze(0)).centers.squeeze(0).unbind()
    clustered_list = []
    for item in temp_list:
        clustered_list.append(item.unsqueeze(0))
    return clustered_list


def normalize(vector_list, text_encoder, tokenizer):
    embs = text_encoder.get_input_embeddings().weight.data
    normalized_vector_list = []
    nearest_tokens = []
    for vector in vector_list:
        values, indices = torch.sort(torch.nan_to_num(torch.nn.functional.cosine_similarity(vector, embs)), dim=0, descending=True)
        nearest = f"Token: {tokenizer._convert_id_to_token(indices[0].item())} sim: {values[0]:.3f}"
        norm = embs[indices[0]].norm()
        normalized_vector_list.append(vector * (norm / vector.norm()))
        nearest_tokens.append(nearest)
    return normalized_vector_list, nearest_tokens


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Embedding Creator')
    parser.add_argument('--model', type=str, help='absolute path to model')
    parser.add_argument('--name', type=str, help='embedding name')
    parser.add_argument('--prompt', type=str,
                        help='positive prompt, comma separation condenses phrases into single tokens')
    parser.add_argument('--negative', type=str, default='',
                        help='negative prompt, comma separation condenses phrases into single tokens')
    parser.add_argument('--maxtokens', type=int, default=0,
                        help='Max number of tokens, uses k-means clustering to reduce if needed')
    args = parser.parse_args()

    pipe = StableDiffusionXLPipeline.from_single_file(args.model)

    prompt = args.prompt.split(",")
    neg_prompt = args.negative.split(",")

    pos_tokens = tokenize(prompt, pipe.tokenizer)
    neg_tokens = tokenize(neg_prompt, pipe.tokenizer)

    pos_vectors_l = get_vectors(pos_tokens, pipe.text_encoder)
    pos_vectors_g = get_vectors(pos_tokens, pipe.text_encoder_2)

    neg_vectors_l = get_vectors(neg_tokens, pipe.text_encoder)
    neg_vectors_g = get_vectors(neg_tokens, pipe.text_encoder_2)

    if neg_vectors_l:
        pos_vectors_l = negate(pos_vectors_l, neg_vectors_l)

    if neg_vectors_g:
        pos_vectors_g = negate(pos_vectors_g, neg_vectors_g)

    if args.maxtokens:
        if args.maxtokens == 1:
            pass
        else:
            kmeans_model = KMeans(n_clusters=args.maxtokens)
            pos_vectors_l = cluster(pos_vectors_l, kmeans_model)
            pos_vectors_g = cluster(pos_vectors_g, kmeans_model)

    pos_vectors_l, nearest_l = normalize(pos_vectors_l, pipe.text_encoder, pipe.tokenizer)
    pos_vectors_g, nearest_g = normalize(pos_vectors_g, pipe.text_encoder_2, pipe.tokenizer)

    print(f"Nearest tokens clip_l: {nearest_l}")
    print(f"Nearest tokens clip_g: {nearest_g}")

    output = {'clip_l': torch.cat(pos_vectors_l),
              'clip_g': torch.cat(pos_vectors_g),
              }

    save_file(output, filename=args.name + ".safetensors")
