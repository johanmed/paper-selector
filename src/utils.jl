"""Module with helper functions"""

module Utils
export preprocess, process_embed, compute_cosm

# Import required modules
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
- Parse documents and extract contents
- Return dictionary and corpus of files and contents
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
- Take a Corpus
- Process in-place contents of StringDocuments in Corpus
- Embed StringDocument contents using specified model
- Return embeddings of cleaned StringDocuments
"""
function process_embed(docs::Corpus, model::PyObject, normalize = true)::Matrix{Float32}
    prepare!(docs, strip_punctuation | strip_numbers | strip_case | strip_whitespace)
    texts = [text(doc) for doc in documents(docs)]
    embeddings = model.encode(texts, normalize_embeddings = normalize)
    return embeddings
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
