import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import pandas as pd

st.set_page_config(page_title="Welcome!", page_icon=":the_horns:", layout='centered',
                   initial_sidebar_state='collapsed', 
                   menu_items={
                    "Get Help":None,
                    "Report a Bug": None,
                    "About":None})

st.markdown("<h1 style='text-align: center;'>Welcome to my webpage!</h1>", unsafe_allow_html=True)

my_image = image.imread("Brunch_Joe_Head_Shot.jpg")

left_column, right_column = st.columns(2)

with left_column:
    st.image(my_image, caption="Me at brunch.", use_column_width=True)

with right_column:
    st.markdown(
    """
    ### Hello!

    My name is Joseph Aaron Gene Diaz and I am a recent graduate of the Applied 
    Mathematics: Emphasis in Dynamical Systems Masters Program at San Diego State 
    University. I am pursuing a career in Data Science and Machine Learning that 
    complements my research and thesis work and utilizes the quantitative skills 
    I've cultivated in my internships and education.
    This webpage is meant to introduce myself to potential employers and assist 
    in my transition from the life of a student to one of a professional.
    """)


tab_prof, tab_skill, tab_mwh, tab_mr = st.tabs(["\U0001F464 Profile", "\U0001F6E0 Skills", 
                                                "\U0001F477 My Work History", 
                                                "\U0001F52C My Research"])

with tab_prof:
    st.markdown(
    """
    ### Under Construction
    """)

with tab_skill:
    st.markdown(
    """
    ### Under Construction
    """)

with tab_mwh:
    st.markdown(
    """
    * ***Graduate Teach Assistant, San Diego State University***
        - __Teaching__: I wrote interactive code lectures that utilized the Python and markdown 
        features of Jupyter Notebooks to provide supplementary instruction in a course on 
        introductory Python programming, data analysis, visualization, and the employment of 
        Python packages such as Numpy, Scipy, Pandas, and Matplotlib.

        - __Tutoring__: I coached my students in any foundational material that was necessary to 
        understand the course content and fill any gaps in knowledge.

        - __Automated Grading__: I made use of the Unit Test class in python to program an automated 
        homework grader on the Gradescope online grading platform.

    ------------

    * ***Student Researcher, San Diego State University Research Foundation***
        - __Machine Learning__: I experimented with neural network architectures in the TensorFlow 
        API to create models to approximate statistical functions and predict future states of 
        dynamical systems.

        - __Data Visualization__: I implemented statistical methods from the statistics submodule 
        of Scipy and Matplotlib to represent and visualize the evolution of Machine Learning models 
        as training takes place with the goal of quanitfying good training when standard metric are 
        unavailable.

    ------------

    * ***Data Analyst/Scientist, Cryogenic Exploitation of Radio Frequency Lab***
        - __Automation__: I used the Socket API in Python to create interfaces for electronic lab 
        equipment to automate the running of experiments and data collection.

        - __Web Programming__: I assisted in the creation of web-based applets for data 
        visualization and analysis by utilizing Python implementations of the QT, Dash, and 
        Streamlit interfaces.

    ------------

    """)

with tab_mr:
    st.markdown(    
    f"""
    #### How do we characterize the evolution of a Machine Learning model under training?

    One of the benefits of neural network models is the ability to learn arbitrary 
    functions from data under training and subject to some loss function. They do this
    by optimizing the weights of the kernel and bias matrices that make up the model
    with respect to that loss function. Consequently even a simple model with a 
    single hidden layer can have several hundred trainable parameters, which in the 
    parlance of dynamical systems means that you have a system with several hundred
    dimensions. Contrast this with something like Lorenz 63
    """)

    st.latex(r"\begin{align*} a &= b+c \\ e &= f-d \\ q &= 3 \end{align*}")
    
    st.markdown(    
    f"""
    which is a 3-dimensional system and we can readily understand it's evolution visually.
    """)
    
    st.markdown(    
    """
    If you're interested in reading my thesis, you can download a PDF copy below: 
    """)
    c = st.columns(3)
    with c[1]:
        with open("A Study on Quantifying Effective Training of DLDMD - Joseph Diaz.pdf", "rb") as file:
            st.download_button("Thesis PDF download.", 
                            data=file,
                            file_name="A Study on Quantifying Effective Training of DLDMD - Joseph Diaz.pdf")





# st.markdown("The graph of $y = \sin(x^a)$ is given below.")

# a = st.slider('\U0001d44e', 1, 10, step=1)

# x = np.linspace(-np.pi, np.pi, 1001)

# fig, ax = plt.subplots(figsize=(15,8))

# ax.plot(x, np.sin(x**a), '-k', label="$y=sin(x^a)$")
# ax.set(xlabel="$x$", ylabel="$y$", title="A SINE WAVE WITH VARIABLE FREQUENCY!!!")
# ax.legend(loc='best')
# ax.grid(True, which='both')

# st.pyplot(fig, clear_figure=True)
# st.caption("The graph that was promised.")



# the_file = st.file_uploader("Dooooo a thing.", type="pkl")
# if the_file is not None and the_file.name.endswith(".pkl"):
    # hyp_params = pkl.load(the_file)
    # for key in hyp_params.keys():
        # st.write(f"{key}: {hyp_params[key]}")
# else:
    # st.write("That is an unsupported file type, please upload a pickle file.")

