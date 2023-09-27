import streamlit as st
import pandas as pd
import base64
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# Page title
st.set_page_config(page_title='Ask the Data App')
st.title('SQL Queries')

# Load CSV file
def load_csv(input_csv):
    df = pd.read_csv(input_csv)
    with st.expander('See DataFrame'):
        st.write(df)
    return df

# Generate LLM response
def generate_response(csv_file, input_query):
    llm = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0.2, openai_api_key=openai_api_key)
    df = load_csv(csv_file)
    # Create Pandas DataFrame Agent
    agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)
    # Perform Query using the Agent
    response = agent.run(input_query)
    return response

# Input widgets
uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])
question_list = [
    'How many rows are there?',
    'What is the range of values for MolWt with logS greater than 0?',
    'How many rows have MolLogP value greater than 0.',
    'Other']
query_text = st.selectbox('Select an example query:', question_list, disabled=not uploaded_file)
openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))

# Add a selectbox to choose the output format
output_format = st.radio('Choose the output format:', ['CSV', 'XLSX/XLS'])

# App logic
if query_text == 'Other':
    query_text = st.text_input('Enter your query:', placeholder='Enter query here ...', disabled=not uploaded_file)
if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
if openai_api_key.startswith('sk-') and (uploaded_file is not None):
    st.header('Output')
    response = generate_response(uploaded_file, query_text)

    # Display the response
    st.subheader('Query Response')
    st.write(response)

    # Add a button to download the response as the selected file format
    st.subheader('Download Response')
    response_bytes = response.encode()
    b64 = base64.b64encode(response_bytes).decode()
    
    if output_format == 'CSV':
        href = f'<a href="data:file/csv;base64,{b64}" download="response.csv">Download Response (CSV)</a>'
    else:
        # Assuming XLSX format if not specified as XLS
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="response.xlsx">Download Response (XLSX)</a>'
    
    st.markdown(href, unsafe_allow_html=True)
