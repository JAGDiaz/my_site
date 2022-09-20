import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
from scipy import integrate
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def lorenz(t,X):
    x, y, z = X
    return np.array([10*(y - x), x*(28 - z) - y, x*y - 8*z/3])

@st.cache
def get_lorenz_traj(t, n=3001):
    times = np.linspace(0, t, n)
    trajs = integrate.solve_ivp(lorenz, times[[0,-1]], [-1, 1, 0], 
                                       t_eval=times, vectorized=True)

    df = pd.DataFrame({"t": trajs.t, **{c: a for c, a in zip("xyz", trajs.y)}})
    df.to_pickle("lorenz63.pkl")

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
    Mathematics Program at San Diego State 
    University. I am pursuing a career in Data Science and Machine Learning that 
    complements my research and thesis work and utilizes the quantitative skills 
    I've cultivated in my internships and education.
    This webpage is meant to introduce myself to potential employers and assist 
    in my transition from the life of a student to one of a professional.
    """)



# tab_prof, tab_skill, tab_mwh, tab_mr = st.tabs(["\U0001F464 Profile", "\U0001F6E0 Skills", 
                                                # "\U0001F477 My Work History", 
                                                # "\U0001F52C My Research"])

with st.expander("\U0001F464 Profile"):
    st.markdown(
    """
    ### A little about me...

    I am a San Diego native who has had 

    ### Under Construction

    """)

# with st.expander("\U0001F6E0 Skills"):
    # st.markdown(
    # """
    # ### Under Construction
    # """)

with st.expander("\U0001F477 My Work History"):

    st.markdown("If you'd like to read my resume, you can download a PDF copy below:")

    c1 = st.columns(3)
    with c1[1]:
        with open("Resume for Joseph Diaz.pdf", "rb") as file:
            st.download_button("Resume PDF download.", 
                            data=file,
                            file_name="Resume for Joseph Diaz.pdf",
                            key="resume_download")

    st.markdown(
    """
    ------------
    
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

    """)

with st.expander("\U0001F52C My Research"):
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
    st.latex(r"""
             \begin{align*}
                \dot{x} &= 10(y - x) \\
                \dot{y} &= x(28 - z) - y \\
                \dot{z} &= xy - 8z/3
             \end{align*}""")
    st.markdown(    
    f"""
    which is a 3-dimensional system of ordinary differential equations and we can readily 
    understand it's evolution visually.
    """)

    # get_lorenz_traj(50, n=5001)

    dataframe = pd.read_pickle("lorenz63.pkl")

    t, x, y, z = dataframe[list("txyz")].values.T
    del dataframe

    fig = go.Figure(
                    data=[
                        go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=t, colorscale='jet'))
                        ],
                    layout=go.Layout(scene=dict(xaxis=dict(range=[x.min(), x.max()], nticks=6), 
                                                yaxis=dict(range=[y.min(), y.max()], nticks=6), 
                                                zaxis=dict(range=[z.min(), z.max()], nticks=5),
                                                camera=dict(up=dict(x=0, y=0, z=1),
                                                            eye=dict(x=-1.25, y=1.25, z=.65,)
                                                            )),
                                    scene_aspectmode='cube'),
                    )

    st.plotly_chart(fig, use_container_width=True)
    del x,y,z,t

    st.markdown(    
    f"""
    While there are tricks to representing dimensions higher 3 they can't cope with several
    hundred dimensions, let alone thousands. As such, we can consider the weights of the matrices
    from a more probablistic point of view: As the network evolves, what is the probability that 
    a given weight appears in the weight matrices for each layer? 

    ### Under Construction
    """)
    
    st.markdown(    
    """
    If you're interested in reading my thesis, you can download a PDF copy below: 
    """)
    c2 = st.columns(3)
    with c2[1]:
        with open("A Study on Quantifying Effective Training of DLDMD - Joseph Diaz.pdf", "rb") as file:
            st.download_button("Thesis PDF download.", 
                            data=file,
                            file_name="A Study on Quantifying Effective Training of DLDMD - Joseph Diaz.pdf",
                            key="thesis_download")





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

