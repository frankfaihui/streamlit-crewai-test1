from openai import OpenAI
import streamlit as st
import os
from dotenv import load_dotenv
from latest_ai_development.crew import LatestAiDevelopment
import datetime
import re

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
                # Create an expander to show thinking steps if desired
                thinking_expander = st.expander("View CrewAI thinking process")
                
                # Collect thinking steps
                thinking_steps = []
                
                # Pass the topic parameter instead of query, and include current_year
                current_year = datetime.datetime.now().year
                
                # Define step callback function
                def step_callback(step_output):
                    if step_output:
                        thought_text = f"### 👤 Agent: {getattr(step_output, 'agent', 'Unknown')}\n"
                        
                        # Get task name or description
                        task_name = getattr(step_output, 'name', None)
                        task_desc = getattr(step_output, 'description', None)
                        
                        if task_name:
                            thought_text += f"**📋 Task Name:** {task_name}\n\n"
                        
                        if task_desc:
                            thought_text += f"**📋 Description:** {task_desc}\n\n"
                        
                        thinking_steps.append(thought_text)
                        with thinking_expander:
                            st.markdown(thought_text)
                
                # Initialize CrewAI with the step callback
                crew_instance = LatestAiDevelopment(step_callback=step_callback)
                
                # Set up a crew
                crew = crew_instance.crew()
                
                # Run CrewAI
                crew_result = crew.kickoff(
                    inputs={"topic": prompt, "current_year": current_year}
                )
                
                # Process the CrewAI response through OpenAI
                if isinstance(crew_result, dict) and "raw" in crew_result:
                    crew_content = crew_result["raw"]
                else:
                    crew_content = str(crew_result)
                
                # Capture thinking steps for the response
                thinking_summary = ""
                if thinking_steps:
                    thinking_summary = "\n\n### 🧠 CrewAI Thinking Process:\n\n" + "\n\n".join(thinking_steps)
                
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
                
                # Add thinking steps if the user requested them
                if "thinking" in prompt.lower() or "steps" in prompt.lower():
                    response = processed_response + thinking_summary
                else:
                    response = processed_response
                
                # Display the processed response
                st.write(response)
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