from ..llms.models import ChatState
from ..llms.engine import llm, structured_llm, structured_llm_memory, structured_llm_answer, mem0
from ..tools.rag import vector_search
from ..tools.web import website_search
from ..utils import memo_id
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def planning(state: ChatState):
   print("Planning node started")

   SYSTEM_PROMPT = """
      You are an Agentic RAG assistant.

      You MUST follow these rules exactly:

         1. If the question is of greeting category then just go to the fallback LLM node to formally reply.
         2. If the question can be asnwered from long term memory then use fall back llm node to formally reply.
         3. Use vector surch node for each and every question.
         4. If the question is not found in the vector store, use web search node.
         5. If the question is not found in the web search, use fall back llm node.
   
      """

   system = SystemMessage(
      content=SYSTEM_PROMPT  
   )
   user = HumanMessage(content=state["question"])

   greeting_classification_prompt = f"""
      Return a JSON object with a single key 'category' that is either 'greeting' or 'non_greeting'. Don't answer the question 
      Analyze if this message is a greeting: {state["question"]}
   """
   classification = structured_llm.invoke(greeting_classification_prompt).category
   print(f"Classification: {classification}")


   memory_classification_prompt = f"""
      Classify the user query into one of these:
         - self_memory: if the user is asking about themselves, their history, projects, identity, or past conversations.
         - general_knowledge: if the question is about people, places, facts, or anything external.

      Return a JSON object with a single key 'category' that is either 'self_memory' or 'general_knowledge'. Don't answer the question.

      Query: {state["question"]}
      """
   memory_classification = structured_llm_memory.invoke(memory_classification_prompt).category
   print(f"Memory classification: {memory_classification}")


   # retreiving long term memory.

   memories = mem0.search(state["question"], filters={"user_id": memo_id})

   # Handle dict response format
   memory_list = memories['results']
   context = "Relevant information from previous conversations:\n"
        
   if len(memory_list) == 0:
      context = "No memory context for similar query found."
   else:
      for memory in memory_list:
         context += f"- {memory['memory']}\n"

   print("CONTEXT FROM LONG TERM MEMORY")
   print(context)

   print("Planning node finised")
   if classification.strip().upper() == "GREETING" or memory_classification.strip().upper() == "SELF_MEMORY":
      return {
         "messages": state["messages"] + [system, user],
         "phase": "fall_back_llm",
         "ltm_context": context,
         "web_context": "",
         "rag_context": ""
      }
   else:
      return {
         "messages": state["messages"] + [system, user],
         "phase": "rag",
         "ltm_context": context,
         "web_context": "",
         "rag_context": ""
      }

def context_search(state: ChatState):
    print("context search started")
        
    context_search = vector_search.invoke(state["question"])

    prompt = f"""
      You are a retrieval-augmented generation (RAG) assistant.

      Your job is to answer the user's question using ONLY the provided context.
      The context comes from a knowledge base and may include multiple documents.

      Rules:
        - If the answer is in the context, use it and cite it implicitly.
        - If the context does NOT contain the answer, say:
          "I don't know based on the provided information."
        - Do NOT use prior knowledge or make up facts.
        - Do NOT hallucinate.
        - Be concise, clear, and factual.
      
      Long Term Memory rules:
        - Do consider long term memory when answering the question about user's personal details, preferences, project or hobbies or anuthing retreived from there.
        - Make the answer user personalized from long term memory. 
      Question:
        {state['question']}

      Context:
        {context_search}

      Long Term Memory:
        {state["ltm_context"]}
      """

    llm_res = llm.invoke(state["messages"] + [HumanMessage(content=prompt)]).content
    
    print("context search finised")
    return {"rag_context": llm_res, "phase": "rag"}

def check_relevence(state: ChatState):
    print("check relevence started")
    
    if state["phase"] == "rag":
        context = state["rag_context"]
    else:
        context = state["web_context"]

    judge_prompt = f"""
    Question: {state['question']}
    Context:
    {context}

    Review the context above and determine if it provides a COMPLETE answer to ALL parts of the user's question: "{state['question']}".
    If ANY part of the question cannot be answered using the provided context, respond with no.
    Only respond with yes if EVERYTHING in the question can be answered using only the provided context.
    Reply only yes or no.

     Return a JSON object with a single key 'category' that is either 'yes' or 'no'. Don't answer the question    
    """

    decision = structured_llm_answer.invoke(judge_prompt).category
    print(f"Decision: {decision}")
    print("check relevance finished")
    if decision.strip().upper() == "YES":
        interaction = [
            {
                "role": "user",
                "content": state["question"]
            },
            {
                "role": "assistant", 
                "content": context
            }
        ]
        result = mem0.add(interaction, user_id=memo_id)
        return {"phase": "end", "messages": [AIMessage(content=context)]}
    else:
        if state["phase"] == "web":
            return {"phase": "fall_back_llm"}
        return {"phase": "web"}

def fall_back_llm(state: ChatState):
    print("fallback node started")
    prompt = f"""

    If the question is knowledge based then: 

    Answer the following question using your knowledge.
    Do NOT use tools.
    Keep the answer high-level, educational and professional.
    Do consider long term memory when asking the question.

    OR if the question is greetings based then: 

    Long Term Memory rules:
        - Do consider long term memory when answering the question about user's personal details, preferences, project or hobbies or anuthing retreived from there.
        - Make the answer user personalized from long term memory. 

    Answer the greetings of the user formally. Don't ever change your tone.

    Question: {state['question']}
    Long Term Memory: {state["ltm_context"]}
    """

    answer = llm.invoke(state["messages"] + [HumanMessage(content=prompt)]).content

    interaction = [
        {
            "role": "user",
            "content": state["question"]
        },
        {
            "role": "assistant", 
            "content": answer
        }
    ]

    result = mem0.add(interaction, user_id=memo_id)

    print("fall back node finished")
    return {
        "messages": [AIMessage(content=answer)],
        "phase": "end"
    }

def web_search(state: ChatState):
    print("web search started")
    
    web_search = website_search.invoke(state["question"])

    prompt = f"""
      You are a retrieval-augmented generation (RAG) assistant.

      Your job is to answer the user's question using ONLY the provided context.
      The context comes from a knowledge base and may include multiple documents.

      Rules:
        - If the answer is in the context, use it and cite it implicitly.
        - If the context does NOT contain the answer, say:
          "I don't know based on the provided information."
        - Do NOT use prior knowledge or make up facts.
        - Do NOT hallucinate.
        - Be concise, clear, and factual.

      Long Term Memory rules:
        - Do consider long term memory when answering the question about user's personal details, preferences, project or hobbies or anuthing retreived from there.
        - Make the answer user personalized from long term memory. 

      Question:
        {state['question']}

      Context:
        {web_search}
      
      Long Term Memory:
        {state["ltm_context"]}
      """
  
    llm_res = llm.invoke(state["messages"] + [HumanMessage(content=prompt)]).content
    
    print("web search finised")
    return {"web_context": llm_res, "phase": "web"}
