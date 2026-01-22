"""Module with helper functions"""

module Utils
export preprocess, process_embed, compute_bm25, compute_cosm

using Languages
using LinearAlgebra
using OrderedCollections
using PDFIO
using PyCall
using SparseArrays
using TextAnalysis

"""
Function that preprocess pdf documents:
- Take directory containing documents in pdf format
- Parse documents and extract contents from pdf
- Return dictionary of files and contents and properly formatted corpus
"""
function preprocess(
    input_path::String,
)::Tuple{Dict{String,String},Corpus{StringDocument{String}}}
    collection = Dict{String,String}()
    files = readdir(input_path, join = true)
    for file in files
        io = IOBuffer()
        doc = pdDocOpen(file)
        npage = pdDocGetPageCount(doc)
        for num = 1:npage
            page = pdDocGetPage(doc, num)
            pdPageExtractText(io, page)
        end
        content = String(take!(io))
        collection[file] = content
    end
    collection = sort(collection) # sort by file for traceback
    corpus = Corpus([
        StringDocument(
            collection[file],
            TextAnalysis.DocumentMetadata(Languages.English(), file),
        ) for file in keys(collection)
    ])
    return collection, corpus
end


"""
Function that completes processing of documents and embeds their contents:
- Take a Corpus of StringDocuments and an embedding model from Python interface
- Process in-place contents of StringDocuments in Corpus
- Embed StringDocument contents using specified embedding model
- Return embeddings of cleaned StringDocuments
"""
function process_embed(
    docs::Corpus{StringDocument{String}},
    model::PyObject,
    normalize = true,
)::Matrix{Float32}
    prepare!(docs, strip_punctuation | strip_numbers | strip_case | strip_whitespace)
    texts = [text(doc) for doc in documents(docs)]
    embeddings = model.encode(texts, normalize_embeddings = normalize)
    return embeddings
end


"""
Function that computes similarity scores of pdf documents to a query based on BM25 search algorithm
- Take a Corpus of StringDocuments (pdf documents) and query StringDocument
- Compute the frequency of each term of the query document by document
- Return normalized BM25 similarity scores
"""
function compute_bm25(
    docs::Corpus{StringDocument{String}},
    query::StringDocument{String},
)::Vector{Float64}
    update_lexicon!(docs)
    dtm_obj = DocumentTermMatrix(docs)
    vocab = dtm_obj.terms
    bm25_mat = bm_25(dtm(dtm_obj); κ = 1, β = 1.0)
    query_tokens = tokens(query)
    query_vec = zeros(length(vocab))
    for token in query_tokens
        idx = findfirst(==(token), vocab)
        if idx !== nothing
            query_vec[idx] += 1.0   # term frequency in query
        end
    end
    bm25_scores = bm25_mat * query_vec
    max_bm = maximum(bm25_scores)
    bm25_norm = max_bm > 0 ? bm25_scores ./ max_bm : zeros(length(bm25_scores))
end

"""
Function that computes cosine similarity between two embedding vectors:
- Take embedding vectors a and b
- Compute and returns cosine similarity value
"""
function compute_cosm(a::AbstractVector, b::AbstractVector)::Float32
    num = dot(a, b)
    den = norm(a) * norm(b)
    return den > 0 ? num / den : 0.0f0
end

end
