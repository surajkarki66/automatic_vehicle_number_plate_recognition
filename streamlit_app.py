import streamlit as st

def main():
    st.title("Basic Streamlit App")
    st.write("Welcome to my simple Streamlit application!")
    
    name = st.text_input("Enter your name:")
    if name:
        st.write(f"Hello, {name}!")
    
    number = st.number_input("Enter a number:", min_value=0, max_value=100, value=50)
    st.write(f"You entered: {number}")
    
    if st.button("Click me!"):
        st.success("Button clicked!")

if __name__ == "__main__":
    main()