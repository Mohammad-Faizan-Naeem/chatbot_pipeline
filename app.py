import streamlit as st
from chatbot_pipeline import chatbot_pipeline  # Ensure this function is correctly imported

def handle_submit():
    user_input = st.session_state.user_query.strip()
    if user_input:
        # Get response from the chatbot pipeline
        response = chatbot_pipeline(user_input)

        # Update current query and response
        st.session_state.current_query = user_input
        st.session_state.current_response = response

        # Add to chat history (latest messages first)
        st.session_state.chat_history.insert(0, ("Bot:", response))
        st.session_state.chat_history.insert(0, ("You:", user_input))

def main():
    # Streamlit UI setup
    st.title("💬 Customer Support Chatbot")
    st.markdown("Ask me anything about our services")

    # Initialize chat history if not already set
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "user_query" not in st.session_state:
        st.session_state.user_query = ""

    if "current_query" not in st.session_state:
        st.session_state.current_query = ""

    if "current_response" not in st.session_state:
        st.session_state.current_response = ""

    # Text input box for user query
    user_query = st.text_input("You:", key="user_query")  # This will handle the input box

    # Button to submit query and get response
    if st.button("Send"):
        handle_submit()  # Only call this function on button click

    # Trigger the same function when Enter is pressed (handle the "on_change" action)
    if st.session_state.user_query.strip() != "" and user_query != st.session_state.user_query:
        handle_submit()

    # Display the latest query and response
    if st.session_state.current_query and st.session_state.current_response:
        st.markdown(
            f"<div style='background-color:#D9ECF2; border-radius: 10px; padding: 10px; text-align: right; color:black; margin-bottom: 10px;'><b>You:</b> {st.session_state.current_query}</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='background-color:#D4EDDA; border-radius: 10px; padding: 10px; text-align: left; color: black; margin-bottom: 10px;'><b>Bot:</b> {st.session_state.current_response}</div>",
            unsafe_allow_html=True
        )

    # Display chat history (latest first)
    st.markdown("<hr>", unsafe_allow_html=True)
    for speaker, message in st.session_state.chat_history[2:]:
        bg_color = "#D9ECF2" if speaker == "You:" else "#D4EDDA"
        text_align = "right" if speaker == "You:" else "left"

        st.markdown(
            f"<div style='background-color:{bg_color}; border-radius: 10px; padding: 10px; text-align: {text_align}; color: black; margin-bottom: 10px;'><b>{speaker}</b> {message}</div>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()

