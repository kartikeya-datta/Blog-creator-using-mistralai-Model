import streamlit as st
from transformers import pipeline
import warnings

# Suppress the warning
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

# Initialize the GPT-2 model for text generation
generator = pipeline('text-generation', model='gpt2')

# Function to generate blog content based on the input topic
def generate_blog(topic: str, max_length=1000):
    """
    Generate blog content based on a given topic.
    
    Args:
    topic (str): The topic or title of the blog.
    max_length (int): Maximum length of the generated blog content.
    
    Returns:
    str: The generated blog content.
    """
    prompt = f"Write a detailed blog about {topic}"
    
    # Generate the content using the model
    generated_content = generator(prompt, max_length=max_length, num_return_sequences=1)
    
    # Extract and return the generated text
    return generated_content[0]['generated_text']

# Main function to run the Streamlit app
def main():
    # Streamlit Web UI
    st.title('Automatic Blog Writing')

    # Input field for blog topic
    topic = st.text_input('Enter a blog topic:', 'How to Learn Machine Learning')

    # Generate button
    if st.button('Generate Blog'):
        if topic:
            blog_content = generate_blog(topic)
            st.subheader('Generated Blog:')
            st.write(blog_content)
        else:
            st.write("Please enter a blog topic.")

# Check if the script is being run directly
if __name__ == "__main__":
    main()