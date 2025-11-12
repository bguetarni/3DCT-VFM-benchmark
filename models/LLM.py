from sentence_transformers import SentenceTransformer

def load(device): 
    model = SentenceTransformer("sentence-transformers/embeddinggemma-300m-medical")
    model.eval().to(device=device)
    infer_ = lambda i: model.encode_document(i)
    return infer_
