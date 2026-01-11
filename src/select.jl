"""
Main script of the program
To run: julia select.jl
"""

using DataFrames
using Utils
import CSV

INPUT_PATH = get(ENV, "INPUT_PATH", "../input")
MODEL_NAME = get(ENV, "MODEL_NAME", "Alibaba-NLP/gte-multilingual-base")
NUM_QUERIES = get(ENV, "NUM_QUERY", 5)
TOP_RESULTS = get(ENV, "TOP_RESULTS", 10)
OUTPUT_PATH = get(ENV, "OUTPUT_PATH", "../output")

"""
Represents a paper selector
"""
struct Selector
    input_path::String
    output_path:String
    model_name::String
    num_queries::Int8
    top_results:Int8
end


"""
Function that extracts embeddings for documents and queries:
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
    query_embs = process_embed(queries, model)

    return collection, queries, query_embeds, doc_embeds
end


"""
Function that evaluates the relevance of a document for each query based on embeddings and writes the results
"""
function evaluate_relevance(
    selector::Selector,
    collection::OrderedCollections.OrderedDict,
    queries::Corpus,
    query_embeds::PyArray,
    doc_embeds::PyArray,
)

    queries = [text(query) for query in documents(queries)]
    dfs = DataFrame[]

    for (query, query_embed) in zip(queries, query_embeds)
        cos = [
            compute_cosm(query_embed, view(doc_embeddings, row, :)) for
            row = 1:size(doc_embeds, 1)
        ] # compute similarity row by row along first dimension
        sorted = sortperm(cos, rev = true)[1:selector.top_results] # get indices of best documents
        scores = cos[sorted]

        # Extract corresponding titles
        keys = collect(keys(ori))
        titles = [keys[el] for el in sorted]

        # Organize results
        df = DataFrame(
            "queries" = [query for query = 1:selector.top_results],
            "titles" = titles,
            "scores" = scores,
        )
        push!(dfs, df)
    end
    CSV.write(selector.output_path, dfs)
end


# Main
selector = Selector(
    input_path = INPUT_PATH,
    output_path = OUTPUT_PATH,
    model_name = MODEL_NAME,
    num_queries = NUM_QUERIES,
    top_results = TOP_RRSULTS,
)

collection, queries, query_embeds, doc_embeds = get_embeddings(selector)
evaluate_relevance(selector, collection, queries, query_embeds, doc_embeds)
