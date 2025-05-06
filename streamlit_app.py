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
                # Initialize CrewAI
                crew_instance = LatestAiDevelopment()
                
                # Create an expander to show thinking steps if desired
                thinking_expander = st.expander("View CrewAI thinking process")
                
                # Collect thinking steps
                thinking_steps = []
                
                # Pass the topic parameter instead of query, and include current_year
                current_year = datetime.datetime.now().year
                
                # Set up a crew with verbose output enabled
                crew = crew_instance.crew()
                
                # Setup a function to capture verbose output
                def verbose_callback(output):
                    if output and isinstance(output, str):
                        # Remove ANSI color codes and formatting
                        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                        clean_output = ansi_escape.sub('', output)
                        
                        # Format specific patterns to make them more readable
                        formatted_output = clean_output
                        
                        # Format "# Agent:" headers
                        formatted_output = re.sub(r'# Agent:\s+(.*?)$', r'### 👤 Agent: \1', formatted_output, flags=re.MULTILINE)
                        
                        # Format "## Task:" headers
                        formatted_output = re.sub(r'## Task:\s+(.*?)$', r'**📋 Task:** \1', formatted_output, flags=re.MULTILINE)
                        
                        # Format "## Final Answer:" headers
                        formatted_output = re.sub(r'## Final Answer:\s*$', r'**✅ Final Answer:**', formatted_output, flags=re.MULTILINE)
                        
                        # Format "Step-by-Step Plan" sections
                        formatted_output = re.sub(r'Step-by-Step Plan', r'**📝 Step-by-Step Plan**', formatted_output)
                        
                        thought_text = formatted_output
                        thinking_steps.append(thought_text)
                        with thinking_expander:
                            st.markdown(thought_text)
                
                # Monkey patch the print function to capture verbose output during crew execution
                import builtins
                original_print = builtins.print
                
                def custom_print(*args, **kwargs):
                    output = " ".join(str(arg) for arg in args)
                    verbose_callback(output)
                    return original_print(*args, **kwargs)
                
                # Replace the print function temporarily
                builtins.print = custom_print
                
                try:
                    # Run CrewAI with verbose output
                    crew_result = crew.kickoff(
                        inputs={"topic": prompt, "current_year": current_year}
                    )
                finally:
                    # Restore the original print function
                    builtins.print = original_print
                
                # Process the CrewAI response through OpenAI
                if isinstance(crew_result, dict) and "raw" in crew_result:
                    crew_content = crew_result["raw"]
                else:
                    crew_content = str(crew_result)
                
                # Capture thinking steps for the response
                thinking_summary = ""
                if thinking_steps:
                    # Join thinking steps but filter out duplicates and empty lines
                    unique_steps = []
                    for step in thinking_steps:
                        if step.strip() and step not in unique_steps:
                            unique_steps.append(step)
                    
                    thinking_summary = "\n\n### 🧠 CrewAI Thinking Process:\n\n" + "\n\n".join(unique_steps)
                
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