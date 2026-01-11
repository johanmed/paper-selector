"""
Main script of the program
To run: julia select.jl
"""

using CSV
using DataFrames
using Languages
using OrderedCollections
using PyCall
using TextAnalysis

include("utils.jl")
using .Utils

INPUT_PATH = get(ENV, "INPUT_PATH", "../input")
OUTPUT_PATH = get(ENV, "OUTPUT_PATH", "../output")
MODEL_NAME = get(ENV, "MODEL_NAME", "Alibaba-NLP/gte-multilingual-base")
NUM_QUERIES = get(ENV, "NUM_QUERIES", 5)
TOP_RESULTS = get(ENV, "TOP_RESULTS", 5)

"""
Represents a paper selector
"""
Base.@kwdef struct Selector
    input_path::String
    output_path::String
    model_name::String
    num_queries::Int8
    top_results::Int8
end


"""
Function that extracts embeddings for documents or queries
"""
function get_embeddings(selector::Selector)
    # Load model
    sentence_transformers = pyimport("sentence_transformers")
    model = sentence_transformers.SentenceTransformer(
        selector.model_name,
        trust_remote_code = true,
    )
    collection, docs = preprocess(selector.input_path)
    doc_embeds = process_embed(docs, model)
    println("Enter $(selector.num_queries) instructions one at a time:")
    queries = Corpus([
        StringDocument(
            readline(),
            TextAnalysis.DocumentMetadata(Languages.English(), "Query $n"),
        ) for n = 1:selector.num_queries
    ])
    query_embeds = process_embed(queries, model)
    return collection, queries, query_embeds, doc_embeds
end


"""
Function that evaluates the relevance of a document for each query based on embeddings and writes the results
"""
function evaluate_relevance(
    selector::Selector,
    collection::Dict{String,String},
    queries::Corpus{StringDocument{String}},
    query_embeds::Matrix{Float32},
    doc_embeds::Matrix{Float32},
)
    query_cont, filename_cont, score_cont = [], [], []

    queries = [text(query) for query in documents(queries)]

    for (query, query_vec) in zip(queries, eachrow(query_embeds))
        cos = [
            compute_cosm(query_vec, view(doc_embeds, row, :)) for
            row = 1:size(doc_embeds, 1)
        ]
        sorted = sortperm(cos, rev = true)[1:selector.top_results] # get indices of best documents
        scores = cos[sorted]

        # Extract corresponding filenames
        files = collect(keys(collection))
        rel_files = [files[el] for el in sorted]

        # Copy relevant documents to separate location
        dest_name = join(split(query)[1:3], '_')
        for file in rel_files
            final_name = "$(selector.output_path)/$(dest_name)/$(basename(file))"
            mkpath(final_name)
            cp(file, final_name, force = true)
        end

        # Store results
        append!(query_cont, [query for ind = 1:selector.top_results])
        append!(filename_cont, rel_files)
        append!(score_cont, scores)

    end

    # Organize and save results
    df = DataFrame(queries = query_cont, filenames = filename_cont, scores = score_cont)
    CSV.write("$(selector.output_path)/report.csv", df)
end


# Main
selector = Selector(
    input_path = INPUT_PATH,
    output_path = OUTPUT_PATH,
    model_name = MODEL_NAME,
    num_queries = NUM_QUERIES,
    top_results = TOP_RESULTS,
)

collection, queries, query_embeds, doc_embeds = get_embeddings(selector)
evaluate_relevance(selector, collection, queries, query_embeds, doc_embeds)
