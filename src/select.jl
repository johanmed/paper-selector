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
    alpha::Float16=0.5
end


"""
Function that generates embeddings for pdf documents and queries
- Take selector instance with its data
- Process and convert pdf documents and queries to Corpus
- Produce embeddings for pdf documents and queries
- Return dictionary of documents, queries, query embeddings and document embeddings
"""
function get_embeddings(selector::Selector)
    # Load model from Python interface
    sentence_transformers = pyimport("sentence_transformers")
    model = sentence_transformers.SentenceTransformer(
        selector.model_name,
        trust_remote_code = true,
    )
    collection, docs = preprocess(selector.input_path)
    doc_embeds = process_embed(docs, model)
    println(
        "Let's digest the papers based on queries.\nEnter $(selector.num_queries) queries you are interested in - one at a time:",
    )
    queries = Corpus([
        StringDocument(
            readline(),
            TextAnalysis.DocumentMetadata(Languages.English(), "Query $n"),
        ) for n = 1:selector.num_queries
    ])
    query_embeds = process_embed(queries, model)
    return queries, query_embeds, docs, doc_embeds, collection
end


"""
Function that evaluates the relevance of a pdf document for each query based on embeddings and writes the results
- Take a selector instance, dictionary of documents, queries, query embeddings and document embeddings
- Compute similarity of pdf documents to each query
- Select the top documents for each query
- Organize and store top results for each query
"""
function evaluate_relevance(
    selector::Selector,
    queries::Corpus{StringDocument{String}},
    query_embeds::Matrix{Float32},
    docs::Corpus{StringDocument{String}},
    doc_embeds::Matrix{Float32},
    collection::Dict{String,String},
)
    query_cont, filename_cont, score_cont = [], [], []

    format_queries = [query for query in documents(queries)]
    text_queries = [text(query) for query in format_queries]

    for (text_query, format_query, query_vec) in
        zip(text_queries, format_queries, eachrow(query_embeds))
        sem_scores = [
            compute_cosm(query_vec, view(doc_embeds, row, :)) for
            row = 1:size(doc_embeds, 1)
        ]
        bm25_scores = compute_bm25(docs, format_query)
        hybrid_scores = selector.alpha .* sem_scores .+ (1 - selector.alpha) .* bm25_scores

        sorted = sortperm(hybrid_scores, rev = true)[1:selector.top_results] # get indices of best documents
        scores = hybrid_scores[sorted]

        # Extract corresponding filenames
        files = collect(keys(collection))
        rel_files = files[sorted]

        # Copy relevant documents to separate location
        dest_name = join(split(text_query)[1:min(3, end)], '_')
        for file in rel_files
            final_name = "$(selector.output_path)/$(dest_name)/$(basename(file))"
            mkpath(final_name)
            cp(file, final_name, force = true)
        end

        # Store results
        append!(query_cont, [text_query for ind = 1:selector.top_results])
        append!(filename_cont, [basename(file) for file in rel_files]) # store only basenames
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

queries, query_embeds, docs, doc_embeds, collection = get_embeddings(selector)
evaluate_relevance(selector, queries, query_embeds, docs, doc_embeds, collection)
