from openai import OpenAI
import streamlit as st
import os
from dotenv import load_dotenv
from latest_ai_development.crew import LatestAiDevelopment
import datetime

load_dotenv()
st.title("ChatGPT-like clone")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Check if prompt indicates we should use CrewAI
        if "crew" in prompt.lower() or "research" in prompt.lower():
            with st.spinner("Running CrewAI..."):
                # Initialize and run the CrewAI
                crew_instance = LatestAiDevelopment()
                # Pass the topic parameter instead of query, and include current_year
                current_year = datetime.datetime.now().year
                crew_result = crew_instance.crew().kickoff(inputs={"topic": prompt, "current_year": current_year})
                
                # Process the CrewAI response through OpenAI
                if isinstance(crew_result, dict) and "raw" in crew_result:
                    crew_content = crew_result["raw"]
                else:
                    crew_content = str(crew_result)
                
                # Send the crew result to OpenAI for processing
                openai_response = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes and formats information in a clear, concise manner."},
                        {"role": "user", "content": f"Please format and present this research information in a user-friendly way: {crew_content}"}
                    ],
                    stream=False
                )
                
                # Get the processed response
                processed_response = openai_response.choices[0].message.content
                
                # Display the processed response
                st.write(processed_response)
                response = processed_response
        else:
            # Use OpenAI for normal chat responses
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})