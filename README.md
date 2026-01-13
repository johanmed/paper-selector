# paper-selector
Tools exist online to make literature review a less daunting endeavour. With the increased adoption of AI, more and more are powered with AI. What if you could further digest selected papers based on dynamic queries from your command-line?
## Description
`paper-selector` is an **AI-powered CLI tool** to filter and prioritize pdf documents, especially research articles based on natural language queries. It uses hybrid search (semantic + keyword search) to help you further digest your papers or documents.

## Setup
`paper-selector` is written in Julia. You must first install Julia on your system. You can refer to Julia documentation to get started.
A number of additional packages is also required to run the tool. Install them by running the following command.
```bash
julia setup.jl
```

## Usage
To use `paper-selector`, you need to prepare your documents or papers in an input directory. We only support PDF for now.
You can specify the path to your input directory and other parameters in your Unix environment with:
```bash
export INPUT_PATH=<actual-path>
export OUTPUT_PATH=<actual-path>
export NUM_QUERIES=<actual-value>
export TOP_RESULTS=<actual-value>
export MODEL_NAME=<huggingface-model-name>
```
If those parameters are not defined, `paper-selector` looks by default for documents in a directory named `input` in the project home directory. Make sure that directory exists and contains your papers. Other parameters are set by default. You don't need to provide them unless you want to have more control.

Once you have everything set up, you can run the tool using the commands below:
```bash
cd src
julia select.jl
```
