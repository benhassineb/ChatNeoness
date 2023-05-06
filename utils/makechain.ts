import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

{context}

Question: {question}
Helpful answer in markdown:`;


const CONDENSE_PROMPT_FR = `En vous basant sur la conversation suivante et la question de suivi, reformulez la question de suivi pour qu'elle puisse être posée indépendamment.

Historique de la conversation :
{chat_history}
Question de suivi : {question}
Question indépendante :`;

const QA_PROMPT_FR = `Vous êtes un assistant AI utile. Utilisez les informations suivantes pour répondre à la question à la fin.
Si vous ne connaissez pas la réponse, veuillez simplement indiquer que vous ne savez pas. NE tentez PAS de donner une réponse inventée.
Si la question n'est pas liée aux informations suivantes, veuillez indiquer poliment que vous êtes programmé pour répondre uniquement aux questions liées aux informations fournies.

{context}

Question : {question}
Réponse utile en markdown :`;


export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 1, // increase temepreature to get more creative answers
    modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT_FR,
      questionGeneratorTemplate: CONDENSE_PROMPT_FR,
      returnSourceDocuments: false, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
