import streamlit as st
from stage_description import ImageDescriptor
from PIL import Image

st.set_page_config(layout="wide")
st.title("Image Description Demo")
STORY = 'You walk into the old basement. It looks like droid laboratory.'
QUESTION = "What are three interesting things I can interact with in this image?"

DEVICE = "cuda:1"


def main():
    if 'pipeline' not in st.session_state:
        with st.spinner(text="Loading models. Please wait..."):
            st.session_state['pipeline'] = ImageDescriptor(device=DEVICE)

    story = st.text_input("storyline", STORY)

    input_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if input_image is not None:
        col1, col2, col3 = st.columns(3)
        image = Image.open(input_image)
        with col1:
            st.image(image, use_column_width=True)
        with col2:
            if st.button("Investigate"):
                with st.spinner(text="Working. Please wait..."):
                    answer = st.session_state['pipeline'].investigate_image(image, story)
                    st.write(answer)

        with col3:
            question = st.text_input("question related to the image", QUESTION)
            if st.button("Ask question"):
                with st.spinner(text="Working. Please wait..."):
                    answer = st.session_state['pipeline'].ask_question(image, question)
                    st.write(answer)


if __name__ == "__main__":
    main()
