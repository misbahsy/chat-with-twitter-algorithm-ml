import { OpenAIChat } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PineconeStore } from 'langchain/vectorstores';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `You are an AI assistant for source code of Twitter's Recommendation Algorithm. You are trained on a repo named The Algorthm ML-that open sources some of the ML models used at Twitter.
  The code for the repo is located at https://github.com/twitter/the-algorithm-ml.
You are given the following extracted parts of a technical summary of files in a codebase and a question. 
Provide a conversational answer with hyperlinks back to GitHub.
You should only use hyperlinks that are explicitly listed in the context. Do NOT make up a hyperlink that is not listed.
Include lots of code examples and links to the code examples, where appropriate.
Assume the reader is a technical person but is not deeply familiar with recommendation algorithms.
Assume the reader does not know anything about how the project is strucuted or which folders/files are provided in the context.
Do not reference the context in your answer. Instead use the context to inform your answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about the recommendation algorith, politely inform them that you are tuned to only answer questions about the recommendation algorithm.
Your answer should be at least 300 words and no more than 500 words.
Do not include information that is not directly relevant to the question, even if the context includes it.
Always include a list of reference links to GitHub from the context. Links should ONLY come from the context.

Question: {question}
=========
{context}
=========
Answer:`,
);

export const makeChain = (
  vectorstore: PineconeStore,
  onTokenStream?: (token: string) => void,
) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAIChat({ temperature: 0 }),
    prompt: CONDENSE_PROMPT,
  });
  const docChain = loadQAChain(
    new OpenAIChat({
      temperature: 0,
      modelName: 'gpt-3.5-turbo', //change this to older versions (e.g. gpt-3.5-turbo) if you don't have access to gpt-4
      streaming: Boolean(onTokenStream),
      callbackManager: onTokenStream
        ? CallbackManager.fromHandlers({
            async handleLLMNewToken(token) {
              onTokenStream(token);
              console.log(token);
            },
          })
        : undefined,
    }),
    { prompt: QA_PROMPT },
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
    returnSourceDocuments: true,
    k: 3, //number of source documents to return
  });
};
