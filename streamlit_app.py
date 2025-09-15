import streamlit as st
import requests
import json

# Set page config
st.set_page_config(
    page_title="Text Generation App",
    page_icon="AI",
    layout="wide"
)

# Title and description
st.title("Text Generation App")
st.markdown("Generate text completions using OpenAI or Hugging Face models")

# Function to query Hugging Face API
def query_huggingface(payload, endpoint_url, api_token=None):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    try:
        response = requests.post(
            endpoint_url,
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Function to query OpenAI API using langchain
def query_openai(prompt, api_key, model_name="gpt-3.5-turbo", max_tokens=100, temperature=0.7):
    try:
        from langchain_openai import OpenAI, ChatOpenAI
        from langchain_core.messages import HumanMessage

        if model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]:
            # Use chat model for GPT-3.5 and GPT-4 chat models
            llm = ChatOpenAI(
                api_key=api_key,
                model=model_name,  # Note: use 'model' not 'model_name' for langchain_openai
                max_tokens=max_tokens,
                temperature=temperature
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            return {"generated_text": response.content}
        else:
            # Use completion model for GPT-3 models and instruct models
            llm = OpenAI(
                api_key=api_key,
                model=model_name,  # Note: use 'model' not 'model_name' for langchain_openai
                max_tokens=max_tokens,
                temperature=temperature
            )
            response = llm.invoke(prompt)
            return {"generated_text": response}
    except ImportError:
        return {"error": "Please install required packages: pip install langchain-openai"}
    except Exception as e:
        return {"error": str(e)}

# Model configuration section
st.subheader("Model Configuration")

# Model selection
model_type = st.selectbox(
    "Select Model Type",
    ["OpenAI Models (GPT-3+)", "Hugging Face Models (GPT-1/GPT-2)"],
    help="Choose between OpenAI models or Hugging Face models"
)

if model_type == "OpenAI Models (GPT-3+)":
    col_config1, col_config2 = st.columns([2, 1])

    with col_config1:
        # OpenAI API Key input
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-xxxxxxxxxxxxx",
            help="Enter your OpenAI API key from https://platform.openai.com/api-keys"
        )

    with col_config2:
        # Model selection for OpenAI
        openai_model = st.selectbox(
            "Select OpenAI Model",
            [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-instruct",
                "davinci-002",  # GPT-3
            ],
            help="Choose the OpenAI model to use (GPT-4, GPT-3.5, or GPT-3)"
        )

    st.info("Using OpenAI models requires an API key. Get yours at [platform.openai.com](https://platform.openai.com/api-keys)")

else:  # Hugging Face Models
    col_config1, col_config2 = st.columns([2, 1])

    with col_config1:
        # Endpoint URL input
        endpoint_url = st.text_input(
            "Hugging Face Endpoint URL",
            placeholder="https://your-endpoint.us-east-1.aws.endpoints.huggingface.cloud",
            help="Enter your Hugging Face endpoint URL"
        )

    with col_config2:
        # API Token input (optional)
        hf_api_token = st.text_input(
            "HF API Token (Optional)",
            type="password",
            placeholder="hf_xxxxxxxxxxxxx",
            help="Enter your Hugging Face API token if required"
        )

    st.info("Using Hugging Face models requires a deployed endpoint. Deploy models at [huggingface.co](https://huggingface.co)")

st.divider()

# Create two columns for input and output
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")

    # Text input area
    user_input = st.text_area(
        "Enter your prompt:",
        value="Which one is larger, 9.11 or 9.9?",
        height=150,
        help="Enter the text you want the model to complete"
    )

    # Advanced parameters in an expander
    with st.expander("Advanced Parameters"):
        max_length = st.slider("Max Length/Tokens", 10, 2000, 500)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)

        if model_type == "Hugging Face Models (GPT-1/GPT-2)":
            top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.1)

    # Generate button
    generate_button = st.button("Generate Text", type="primary", use_container_width=True)

with col2:
    st.subheader("Output")

    # Create a placeholder for the output
    output_container = st.container()

    if generate_button:
        if not user_input:
            st.warning("Please enter some text to generate completions")
        elif model_type == "OpenAI Models (GPT-3+)":
            if not openai_api_key:
                st.error("Please enter your OpenAI API key")
            else:
                with st.spinner("Generating text with OpenAI..."):
                    # Call OpenAI API
                    result = query_openai(
                        prompt=user_input,
                        api_key=openai_api_key,
                        model_name=openai_model,
                        max_tokens=max_length,
                        temperature=temperature
                    )

                    # Display the result
                    with output_container:
                        if "error" in result:
                            st.error(f"Error: {result['error']}")
                        else:
                            st.success("Generated Text:")
                            st.write(result["generated_text"])

                            # Show raw response in expander
                            with st.expander("View Raw Response"):
                                st.json(result)

        else:  # Hugging Face Models
            if not endpoint_url:
                st.error("Please enter a Hugging Face endpoint URL")
            else:
                with st.spinner("Generating text with Hugging Face..."):
                    # Prepare the payload
                    payload = {
                        "inputs": user_input,
                        "parameters": {
                            "max_length": max_length,
                            "temperature": temperature,
                            "top_p": top_p
                        }
                    }

                    # Make the API call
                    result = query_huggingface(payload, endpoint_url, hf_api_token if 'hf_api_token' in locals() else None)

                    # Display the result
                    with output_container:
                        if "error" in result:
                            st.error(f"Error: {result['error']}")
                            if "paused" in str(result.get("error", "")).lower():
                                st.info("The endpoint is paused. Please restart it in your Hugging Face account.")
                        else:
                            # Handle different response formats
                            if isinstance(result, list) and len(result) > 0:
                                if "generated_text" in result[0]:
                                    st.success("Generated Text:")
                                    st.write(result[0]["generated_text"])
                                else:
                                    st.json(result)
                            elif isinstance(result, dict):
                                if "generated_text" in result:
                                    st.success("Generated Text:")
                                    st.write(result["generated_text"])
                                else:
                                    st.json(result)
                            else:
                                st.json(result)

                            # Show raw response in expander
                            with st.expander("View Raw Response"):
                                st.json(result)

# Add information in the sidebar
with st.sidebar:
    st.header("Setup Instructions")

    with st.expander("OpenAI Setup"):
        st.markdown(
            """
            **Getting your OpenAI API Key:**
            1. Go to [platform.openai.com](https://platform.openai.com)
            2. Sign up or log in
            3. Navigate to API Keys section
            4. Create a new API key
            5. Copy and paste it in the app

            **Available Models:**
            - **GPT-4o / GPT-4o-mini**: Latest models
            - **GPT-4 / GPT-4-turbo**: Most capable
            - **GPT-3.5-turbo**: Fast and efficient
            - **text-davinci-003**: GPT-3 completion model
            """
        )

    with st.expander("Hugging Face Setup"):
        st.markdown(
            """
            **Getting your endpoint URL:**
            1. Go to [huggingface.co](https://huggingface.co)
            2. Deploy your model as an endpoint
            3. Copy the endpoint URL
            4. Paste it in the app

            **Supported Models:**
            - GPT-1
            - GPT-2 and variants
            - Other Hugging Face text generation models

            **Note:** If your endpoint is paused, restart it from your HF dashboard.
            """
        )

    st.divider()

    # Test connection button
    if st.button("Test Connection"):
        with st.spinner("Testing connection..."):
            if model_type == "OpenAI Models (GPT-3+)":
                if not openai_api_key:
                    st.error("Please enter an OpenAI API key first")
                else:
                    test_result = query_openai(
                        prompt="Hello",
                        api_key=openai_api_key,
                        model_name=openai_model,
                        max_tokens=10,
                        temperature=0.7
                    )
                    if "error" in test_result:
                        st.error(f"Connection failed: {test_result['error']}")
                    else:
                        st.success("OpenAI API connection successful!")
            else:
                if not endpoint_url:
                    st.error("Please enter an endpoint URL first")
                else:
                    test_payload = {
                        "inputs": "Hello",
                        "parameters": {}
                    }
                    test_result = query_huggingface(test_payload, endpoint_url, hf_api_token if 'hf_api_token' in locals() else None)
                    if "error" in test_result:
                        st.error(f"Connection failed: {test_result['error']}")
                    else:
                        st.success("Hugging Face API connection successful!")

    st.divider()

    st.info(
        """
        **Required Libraries:**
        ```bash
        pip install streamlit requests
        pip install langchain-openai  # For OpenAI models
        ```
        """
    )