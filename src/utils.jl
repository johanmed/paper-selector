"""Module with helper functions"""

module Utils

# Import required modules
using PDFI
using TextAnalysis
using LinearAlgebra
using Languages
using PyCall
using SparseArrays

"""
Function that preprocess pdf documents:
- Takes directory containing documents in pdf format
- Parses papers and extract title and body
- Returns dictionary of papers with title as key and body as value
"""
function preprocess(dir_path::String)
    collection = Dict{String,String}()
    files = readdir(dir_path)
    title = ""
    for file in files
        doc = pdDocOpen(file)
        number_page = pdDocGetPageCount(doc)
        for page in number_page
            ref = pdDocGetPage(doc, page)
            io = IOBuffer()
            pdPageExtractText(io, ref)
            content = String(take!(io))
            indices = findfirst("introduction", lowercase(content))
            if indices isa UnitRange
                global title
                title = strip(content[begin:indices[begin-1]]) # take care of title cleaning separately
                other = content[indices[begin]:end]
            else
                other = content
            end
            if length(title) != 0
                if title
                    not in collections
                    collection[title] = other
                else
                    collection[title] *= other
                end
            end
        end
    end

    collection = sort(collection) # sort by key to facilitate traceback
    corpus = Corpus([
        StringDocument(
            collection[title],
            TextAnalysis.DocumentMetadata(Languages.English(), title),
        ) for title in collection
    ])

    return collection, corpus
end


"""
Function that completes processing of documents and embeds their contents:
- Takes a Corpus
- Process in-place contents of StringDocument(s) in Corpus
- Embeds StringDocument contents using specified model
- Returns embeddings of cleaned StringDocument(s)
"""
function process_embed(docs::Corpus, model::PyObject, normalize = true)
    prepare!(docs, strip_punctuation | strip_numbers | strip_case | strip_whitespace)
    texts = [text(doc) for doc in documents(docs)]
    embeddings = model.encode(texts, normalize_embeddings = normalize)
    return PyArray(embeddings)
end


"""
Function that computes cosine similarity between two embedding vectors:
- Takes embedding vectors a and b
- Computes and returns cosine similarity measure
"""
function compute_cosm(a::AbstractVector, b::AbstractVector)
    num = dot(a, b)
    den = norm(a) * norm(b)
    return den > 0 ? num / den : 0.0f0
end


end
