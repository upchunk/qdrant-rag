from llama_index.core.evaluation import (
    AnswerRelevancyEvaluator,
    ContextRelevancyEvaluator,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore


async def relevancy_evaluator(
    llm: FunctionCallingLLM,
    query_str: str,
    contexts: list[str] = None,
    response_str: str = None,
    verbose: bool = False,
):
    if response_str:
        answer_eval_result = await AnswerRelevancyEvaluator(llm=llm).aevaluate(
            query=query_str, response=response_str, contexts=contexts
        )
    if contexts:
        context_eval_result = await ContextRelevancyEvaluator(llm=llm).aevaluate(
            query=query_str, response=response_str, contexts=contexts
        )

    print("##### EVALUATION RESULTS #####")
    if verbose:
        if response_str:
            print(
                "Answer Relevancy Test:\n", answer_eval_result.model_dump_json(indent=2)
            )
        if contexts:
            print(
                "Context Relevancy Test:\n",
                context_eval_result.model_dump_json(indent=2),
            )
    else:
        if response_str:
            print("Answer Relevancy Test:", answer_eval_result.score)
        if contexts:
            print("Context Relevancy Test:", context_eval_result.score)
    print("##############################")


async def retriever_evaluator(
    llm: FunctionCallingLLM,
    retriever: BaseRetriever,
    query_str: str,
    verbose: bool = False,
):
    node_with_scores: list[NodeWithScore] = await retriever.aretrieve(query_str)
    eval_result = await ContextRelevancyEvaluator(llm=llm).aevaluate(
        query=query_str, contexts=[each.get_content() for each in node_with_scores]
    )

    print("##### EVALUATION RESULTS #####")
    if verbose:
        print("Context Relevancy Test:\n", eval_result.model_dump_json(indent=2))
    else:
        print("Context Relevancy Test:", eval_result.score)
    print("##############################")
