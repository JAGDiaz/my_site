import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import matplotlib.animation as anime
from scipy import integrate
import pandas as pd

def lorenz(t,X):
    x, y, z = X
    return np.array([10*(y - x), x*(28 - z) - y, x*y - 8*z/3])

@st.cache
def get_lorenz_traj(t, n=3001, seconds=15):
    times = np.linspace(0, t, n)
    trajs = integrate.solve_ivp(lorenz, times[[0,-1]], [-1, 1, 0], 
                                       t_eval=times, vectorized=True)
 
    df = pd.DataFrame({"t": trajs.t, **{c: a for c, a in zip("xyz", trajs.y)}})
    df.to_pickle("lorenz63.pkl")

    x, y, z = trajs.y
    fig = plt.figure(figsize=(10,7))

    ax = fig.add_subplot(111, projection='3d')
    ax.set(xlim=(x.min(), x.max()), ylim=(y.min(), y.max()), 
           zlim=(z.min(), z.max()),)
    ax.set_xlabel("$x$", size=15)
    ax.set_ylabel("$y$", size=15)
    ax.set_zlabel("$z$", size=15)
    ax.set_title(f"Lorenz Evolution, $t = {times[0]:05.3f}$", size=25)

    lines, = ax.plot(x,y,z, color="tab:red")
    fig.tight_layout()
    
    meta = dict(title=f"Lorenz Evolution", 
                artist="Matplotlib")
    writer = anime.FFMpegWriter(fps=times.size//seconds, metadata=meta)

    with writer.saving(fig, "lorenz63.mp4", 150):

        for ii, t in enumerate(times):
            low_index = max([0, ii-100])
            
            lines.set_data_3d(x[low_index:ii], y[low_index:ii], z[low_index:ii])
            ax.set_title(f"Lorenz Evolution, $t = {t:05.3f}$", size=25)

            writer.grab_frame()

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

    I am a San Diego native who has always been interested in computers and what they
    can do. As such, I originally pursued a degree in Computer Science before becoming
    fascinated with Mathematics and it's intersection with programming and Computational
    Science.

    I am a team player who is also comfortable working independently. I 
    have demonstrated these qualities throughout my college career through academics and 
    community service. They are also demonstrated by my work ethic and interpersonal 
    relationships. I am flexible and versatile, and can maintain a sense of humor while 
    working under pressure.

    """)
    
    # """
    
    # ### A little about me...

    # I am a San Diego native who has always been interested in computers and what they
    # can do. As such, I originally pursued a degree in Computer Science before becoming
    # fascinated with Mathematics and it's intersection with programming and Computational
    # Science.

    # I am a team player who is also comfortable working independently. I 
    # have demonstrated these qualities throughout my college career through academics and 
    # community service. They are also demonstrated by my work ethic and interpersonal 
    # relationships. I am flexible and versatile, and can maintain a sense of humor while 
    # working under pressure.

    # """


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

    * ***Data Analyst/Scientist Intern, Cryogenic Exploitation of Radio Frequency Lab***
        - __Automation__: I used the Socket API in Python to create interfaces for electronic lab 
        equipment to automate the running of experiments and data collection in a 
        data pipeline through the local lab network.

        - __Web Programming__: I assisted in the creation of web-based applets for data 
        visualization and analysis by utilizing Python implementations of the QT, Dash, and 
        Streamlit interfaces through the local lab network.

        - __Data Processing__: I created scripts for identifying, selecting, and 
        replacing or cleaning missing or noisy data points in data sets gleaned from experimental 
        measurements. These scripts also restructured the data to json, hdf5, csv, or added it to a 
        relational database as the project required.

        - __Data Analysis__: I utilized statistical techniques such as smoothing, binning, 
        time series analysis, regression analysis and stochastic analysis to find trends in data sets 
        and adjust models accordingly. These were implemented in conjunction with the 
        automated data collection and data processing code for a seamless pipeline between experiments
        and actionable insights.

        - __Signal Processing__: I operated signal processing equipment for network analysis,
        spectrum analysis, noise-figure characterization, and arbitrary waveform generation, and 
        bolstered efforts to use these for testing the efficacy of experimental electrical circuits.

    ------------

    * ***Graduate Teach Assistant, San Diego State University***
        - __Teaching__: I wrote interactive code lectures that utilized the Python and markdown 
        features of Jupyter Notebooks to provide supplementary instruction in a course on 
        introductory Python programming, data analysis, visualization, and the employment of 
        Python packages such as Numpy, Scipy, Pandas, and Matplotlib.

        - __Tutoring__: I coached my students in foundational material required
        to understand critical course content in addition to clarifying misconceptions and 
        supplementing student's learning by tying new material to the foundational material.

        - __Automated Grading__: I leveraged the Python Unit Test framework to develop and
        deploy an automated homework grader compatible with the Gradescope online grading platform.

    ------------

    * ***Student Researcher, San Diego State University Research Foundation***
        - __Machine Learning__: I explored and investigated neural network architectures in 
        the TensorFlow API to create sequential network models to approximate statistical functions and predict
        future states of dynamical systems. I examined applying well understood statistical methods 
        such as fittings and density estimation to the evolution of individual layers in a neural 
        network model; new networks models with different hyperparamers were developed from this 
        analysis and progress was catalogued for review by my research advisor.

        - __Data Visualization__: I conceived of and wrote algorithms that used the statistics
        submodules of Scipy and Numpy to represent and visualize the evolution of Machine Learning 
        models as training takes place with the goal of quantifying good training when standard 
        metric are unavailable. I translated R code for visualizing climate data into Python 
        using Matplotlib as supplementary material for a course on climate statistics.
    """)

with st.expander("\U0001F52C My Research"):
    
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
    st.markdown(    
    f"""
    ### How do we characterize the evolution of a Machine Learning model under training?

    One of the benefits of neural network models is the ability to learn arbitrary 
    functions from data under training and subject to some loss function. They do this
    by optimizing the weights of the kernel and bias matrices that make up the model
    with respect to that loss function. Consequently even a simple model with a 
    single hidden layer can have several hundred trainable parameters, which in the 
    parlance of dynamical systems means that you have a system with several hundred
    dimensions. Contrast this with something like Lorenz 63
    """)
    st.latex(
    r"""
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

    st.video(open("lorenz63.mp4", 'rb').read(), )

    st.markdown(    
    f"""
    While there are tricks to representing dimensions higher than 3 they can't cope with several
    hundred dimensions, let alone thousands. As such, we can consider the weights of the matrices
    from a more probablistic point of view: As the network evolves, what is the probability that 
    a given weight appears in the weight matrices for each layer? 

    To do this, we generate histograms from each layer and for each epoch. 

    ### Under Construction
    """)

    st.video(open("layer_Output_evolution_hist_.mp4", 'rb').read(), )

