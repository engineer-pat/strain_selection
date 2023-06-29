"""
Streamlit version of the initial notebook here:
https://github.com/patimus-prime/ML_notebooks/blob/master/high-throughput-analysis.ipynb
"""

import streamlit as st
import pandas as pd
import matplotlib as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from datasets import load_dataset
import plotly.express as px

# https://docs.streamlit.io/library/components/components-api
import streamlit.components.v1 as components

# And some extras to have fun with this go:
# I got weird errors, 28 June 2023
# pip install streamlit-extras
# pip install streamlit-elements
#

st.image(
    "memes/sneakpeek.png",
)

colA, colB = st.columns(2)

with colA:
    st.write(
        """
            # Welcome!
            
            Objective: In an experiment with multiple machines running
            separate but similar instruments, prone to inefficiency, or with
            unstable samples, how can you identify statistically significant 
            samples? In this case study, we are searching for samples, strains,
            that are producing the most amount of target molecule.
            
            We'll use some fun statistics to identify prime candidates, and
            Streamlit to make everything beautiful and relatively interactive.        
            
            """
    )


with colB:
    components.iframe(
        "https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/playlists/1644215929&color=%23ff5500\
                    &auto_play=true\
                    &hide_related=false\
                    &show_comments=false\
                    &show_user=false\
                    &show_reposts=false\
                    &show_teaser=false\
                    &visual=false\
                    &buying=false",
        width=int(166 * 2),
        height=int(166 * 1.5),
    )

st.write(
    """
         Load in our data from Huggingface, available here: https://huggingface.co/patimus-prime
         
         ---
         """
)

# datasets from huggingface assume train data
# https://huggingface.co/docs/datasets/load_hub
with st.echo():
    dataset = load_dataset("patimus-prime/strain_selection", split="train")
    df = pd.DataFrame(dataset)

# if you don't specify split="train" above, must do the coupe following lines
# view dataset, from wher which we can se we must grab one of hte indeces for data
# dataset["train"]
# df = pd.DataFrame(dataset['train'])

# Modify stacker_postion to something less nebulous
df["time_mins"] = df["stacker_position"].apply(lambda x: 3 * x - 3)

# st.dataframe(df)
st.write(
    "Let's take a look at the first few rows...",
    df.head(),
    "...and the 'describe' method, which has FORESHADOWING",
    df.describe(),
    "And, plotting the data of course:",
)

fig = px.scatter(
    df,
    x="time_mins",
    y="value",
    # size="pop",
    color="well_type",
    hover_name="strain",
    hover_data="value",
    symbol="robot",
    # log_x=True,
    # size_max=60,
)
st.plotly_chart(fig, use_container_width=True)

st.write(
    """
    A couple things become apparent:
    1. We're going to need to resolve the data being from multiple machines.
    2. Values are trending downward with time/stacker position. Therefore, options:
        time-series analysis, analyze per t-point, or REMOVE THE INFLUENCE OF TIME 👹  
    3. Plotly + Streamlit = Awesome
    
    Our process control is also decreasing with time; this, as a control, should
    remain constant. Suggests that overtime, either the control and
    samples are degrading in step or the instrument is affected. 
    
    So: detrend the data, prove robots comparable with ANOVA, profit.
    
    # Detrend the data
    Assumption: The robots are comparable.
    
    Let's detrend first to be able to pool all day per time point,
    then we can have more statistical OOMPH to do ANOVA and
    demonstrate the robots may be pooled also and back up our assumption. 
    
    Among the options to detrend, e.g. subtracting a rolling average/statistic,
    I choose subtracting a line of best fit. The math is more transparent. And
    we can fit a line of best fit per well type; in case, perhaps,
    the different strain types do have some ongoing, unique phenomenon decaying their value.
    """
)

# note streamlit will use width to also adjust height
st.image("memes/zapp.jpg", width=300)

st.write(
    """
         Big bonus for Streamlit:
         Big improvement over using matplotlib/seaborn:
         plotly express allows one to use OLS fits and also report the 
         stats. Based on what's coming out I'd guess they use statsmodels.
         https://plotly.com/python/linear-fits/
         Oh, maybe you'd like to see how LUXURIOUS the code is for this
         """
)

with st.echo():
    fig2 = px.scatter(
        df,
        x="time_mins",
        y="value",
        color="well_type",
        hover_name="strain",
        hover_data="value",
        symbol="robot",
        trendline="ols",
    )

    st.plotly_chart(fig2)

    stats = px.get_trendline_results(fig2)
    st.write(stats.px_fit_results.iloc[0].summary())

st.write(
    """
        ---
        
         Hit the Home button in the graph if you lose the trendlines.
         Also you can hit labels in legend to toggle on/off on graph.
          
         Pretty cool. However, the trendline summary
         produced below the graph is not really helpful in this case,
         the trendlines produced are at least visually similar,
         but what would be helpful for our math
         is if the trendlines individually per well type
         are similar enough to subtract and not miss out on some 
         phenomenon occurring per well. 
         
         "Just because you're paranoid, doesn't mean they aren't after you."
         -- Joseph Heller
         """
)


# This next section in the code is to get the OLS trendlines
# for each well. Maybe with some trickery you could do it with the plotly stuff.

pc_data = df[df.well_type == "Process Control"]
pc_ols = smf.ols("value ~ time_mins", data=pc_data).fit()
pcs = pc_ols.summary()

ps_data = df[df.well_type == "Parent Strain"]
ps_ols = smf.ols("value ~ time_mins", data=ps_data).fit()
pss = ps_ols.summary()

x_data = df[df.well_type == "Standard Well"]
x_ols = smf.ols("value ~ time_mins", data=x_data).fit()
xs = x_ols.summary()

# A chance to see these summares side by side
# using streamlit columns, make everything pretty pretty prettcol1,
st.write(
    "Feel free to skip past the tables.\
        What we're looking for is the second row for coef, the slope of the lines."
)


if "button" not in st.session_state:
    st.session_state.button = False


def click_button():
    st.session_state.button = not st.session_state.button


st.button("Show delicious statistics", on_click=click_button)

if st.session_state.button:
    st.header("Process Control Summary")
    pcs
    st.header("Parent Strain Summary")
    pss
    st.header("Candidate Strains Summary")
    xs

else:
    st.write("")


st.write(
    """
    So: PC, -26, PS, -33, Candidates, -33.
    Not perfect, and not much we can do but assume we can go on 
    in this analysis, while NOTING!! that this process control and our
    strains, control and candidates, are being treated differently.
    It may well be there's some actual degradation occurring over the course
    of the experiment, maybe target molecule is being measured while being enzymatically
    degraded, or... yeah, just something to investigate in future.
    
    # ANYWAY
    Now we'll begin detrending; that is, subtracting the predicted, trendline value,
    from the actual measured value. Result: our means go to 0 and all detrended values
    become a measure of distance from the mean.
    
    Let's also see the exact code going on:
    """
)

# DETRENDING!! ---------

# PC (and note lambda x is per entry, agnostic, not to be confused with x_data below)
pc_data["predicted_value"] = pc_ols.predict(pc_data.time_mins)
pc_data["detrended_value"] = pc_data.apply(
    lambda x: x["value"] - x["predicted_value"], axis=1
)
# same thing for the other dataframes. hope the math is clear atm!
# PS:
ps_data["predicted_value"] = ps_ols.predict(ps_data.time_mins)
ps_data["detrended_value"] = ps_data.apply(
    lambda x: x["value"] - x["predicted_value"], axis=1
)

with st.echo():
    # X:
    x_data["predicted_value"] = x_ols.predict(x_data.time_mins)
    x_data["detrended_value"] = x_data.apply(
        lambda x: x["value"] - x["predicted_value"], axis=1
    )

st.write(
    "And, so, behold! Scrolling over to detrended value, mean is 0:",
    x_data.describe(),
    "# SO! Now pooling our data and re-plotting...",
)

# get the band back together
# d for detrended
# Aloso pandas has updated:
# https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html

framesToMerge = [pc_data, ps_data, x_data]
d_df = pd.concat(framesToMerge)

fig3 = px.scatter(
    d_df,
    x="time_mins",
    y="detrended_value",
    color="well_type",
    hover_name="strain",
    hover_data="detrended_value",
    symbol="robot",
    # trendline="ols",
    opacity=0.3,
)

st.plotly_chart(fig3)


# For only superior seaborn plot
# maybe come back and try
# s = sns.lineplot(
#     data=d_df,
#     # literally x and y axes
#     x="time_mins",
#     y="detrended_value",
#     # grouping variables:
#     hue="well_type",
#     style="robot",
#     # legend = 'auto', # auto by default
# )
# st.pyplot(s)


st.write(
    """
         This is one case where there's a superior plot in seaborn, lineplot,
         that doesn't let the Standard Wells overwhelm everything. I've
         tried to make it transparent, but one can still
         check, just toggle off the Standard Wells in the legend.
         
         Anyway, the mean is 0, artificially. Yay! We have eliminated time. Yet we still
         are slicing by robot. We will now resolve this and validate our assumption
         that we could compare these things.
         
         Once that's done, we'll be able to consider just 3 populations, 
         not sliced by time or machine:
         Process Control, Parent Strain and the Candidate Strains. 
         """
)

st.image("memes/farnsworthquantum.jpg", width=300)

st.write(
    """
         # ANOVA on Process Controls
         Here we'll do ANOVA on the process controls, now liberated of time.
         By comparing them across machines, and proving there is no statistical difference
         of the populations, we can treat them as one, our method is valid, and assume the rest of the 
         sample types may also be pooled.
         
         A problem, though (which has a solution): ANOVA assumes homoscedasticity,
         It demands the stdev of all groups are the same. The mean is artifically 0 for 
         all populations, but stdev is not likely to be similar and we'll look at it in a second.
         
         An alternative would be repeated measure ANOVA, and do ANOVA per well type
         per time point; but this will have much less statistical power splitting up the data in 20 parts.
         
         Therefore: note the stdevs and go with the lesser evil of regular ANOVA.
         """
)

with st.echo():
    d_pc_df = d_df[d_df.well_type == "Process Control"]

    overall_std = d_pc_df.detrended_value.std()
    bender_std = d_pc_df[d_pc_df.robot == "Bender"].detrended_value.std()
    c3p0_std = d_pc_df[d_pc_df.robot == "C3P0"].detrended_value.std()
    term_std = d_pc_df[d_pc_df.robot == "Terminator"].detrended_value.std()

st.write(
    "stdevs: \n",
    "overall",
    overall_std,
    "bender:",
    bender_std,
    "c3p0:",
    c3p0_std,
    "terminator:",
    term_std,
)

st.write(
    """
         Numbers are fairly similar, although the terminator robot isn't as close as the others.
         We'll assume homoscedasticity.
         The overall stdev of 212 for the process control can be used to define
         the outlier threshold: 3*212 = 636, so anything outside +/- 630 can be considered outliers. 
         Let's peek and see if any exist in the process control population:
         """,
    d_pc_df[d_pc_df.well_type == "Process Control"].describe(),
    """
    Min and max are -556 and 601; seemingly no outliers, nothing beyond the cutoff. 
    So, we can consider this as som analagous measure of the instrument error for all data:
    therefore all other outliers' threshold, for the parent and candidate strains, should be >= 630.
    
    Therefore, let's now confirm the null hypothesis, that all populations are equivalent, for the process
    controls via ANOVA. """,
)

# THere appears to be an alternative method to do this via:
# https://www.statsmodels.org/stable/anova.html
# BUt for now I'll do the scipy option to be lazy

# SCIPY method
# ANOVA!!

from scipy.stats import f_oneway

# the scipy implementation takes in a 1D vector of values... soooo... let's see how to get that out
# d_df[d_df.robot == 'Terminator'].detrended_value.values # this should work i

st.write(
    f_oneway(
        d_pc_df[d_pc_df.robot == "Terminator"].detrended_value.values,
        d_pc_df[d_pc_df.robot == "Bender"].detrended_value.values,
        d_pc_df[d_pc_df.robot == "C3P0"].detrended_value.values,
    ),
    """
    Pvalue is above 0.05; without even examining the F-statistic we can say
    the null hyothesis cannot be rejected, process controls are not statsitically
    significant, not meaningfully differente from each other.
    
    It is however surprising that the p-value isn't much higher given how similar 
    we've seen the process controls look. LIVIN' ON A PRAYER
    
    So, moving forward with assumption that process controls are equivalent, and
    thus we may also compare the other sample types, parent strain and candidate strain.
    
    ... Also, the parent strain may serve as almost a secondary control, so we'll also be looking
    at that in a second.
    """,
)

# https://plotly.com/python/histograms/
# https://plotly.com/python-api-reference/generated/plotly.express.histogram.html?highlight=histogram
fig4a = px.histogram(
    d_df[d_df.well_type == "Process Control"],
    x="detrended_value",
    color="robot",
    hover_name="robot",
    hover_data="detrended_value",
    marginal="violin",  # in margin, visual of dist/data
    opacity=0.5,
    barmode="overlay",
    histfunc="count",
    histnorm="density",
    nbins=50,
)

fig4b = sns.displot(
    data=d_df[d_df.well_type == "Process Control"],
    x="detrended_value",
    hue="robot",
    kind="kde",
)

st.write(
    """
    Nonideal in plotly express. Cannot see density plots that
    make it a bit easier, informative to compare; so density plot in second
    tab. (Warning: seaborn density plot has no dark mode.)
    """
)


tab1, tab2 = st.tabs(["Plotly Histogram", "SNS Density Plot"])

with tab1:
    st.plotly_chart(fig4a)

with tab2:
    st.pyplot(fig4b)

st.write(
    """
         We can see that the Terminator robot is the most consistent, least stdev/
         most populous around the mean. I hope the new movie is good. 
         
         Similarly, let's compare the Parent Strain and Candidate strain sample populations
         per robot. Same tabs etc.
         """
)

fig5a = px.histogram(
    d_df[d_df.well_type == "Parent Strain"],
    x="detrended_value",
    color="robot",
    hover_name="robot",
    hover_data="detrended_value",
    marginal="violin",  # in margin, visual of dist/data
    opacity=0.5,
    barmode="overlay",
    histfunc="count",
    histnorm="density",
    nbins=50,
)

fig5b = sns.displot(
    data=d_df[d_df.well_type == "Parent Strain"],
    x="detrended_value",
    hue="robot",
    kind="kde",
)


fig5a = px.histogram(
    d_df[d_df.well_type == "Standard Well"],
    x="detrended_value",
    color="robot",
    hover_name="robot",
    hover_data="detrended_value",
    marginal="violin",  # in margin, visual of dist/data
    opacity=0.5,
    barmode="overlay",
    histfunc="count",
    histnorm="density",
    nbins=50,
)

fig5b = sns.displot(
    data=d_df[d_df.well_type == "Standard Well"],
    x="detrended_value",
    hue="robot",
    kind="kde",
)

# PARENT STRAIN TABS ------
st.write("Parent Strain (~ second control):")

tab1, tab2 = st.tabs(["Plotly Histogram", "SNS Density Plot"])

with tab1:
    st.plotly_chart(fig4a)

with tab2:
    st.pyplot(fig4b)

# CANDIDATE STAIN TABS ----
st.write("Standard Well/Candidate Strains:")

tab1, tab2 = st.tabs(["Plotly Histogram", "SNS Density Plot"])

with tab1:
    st.plotly_chart(fig5a)

with tab2:
    st.pyplot(fig5b)


st.write(
    """
    
    # ALRIGHTY THEN, CONCLUDING POOLING: 
    
         Alrighty, from these two above we see, *looking at the axes*,
         that the Standard Wells have far more data spread and variabiltiy. The
         process control and parent strain samples fall mostly within +/- 500.
         Versus the Standard Wells going out to like +/- 1500.
         And I'll say now these are ~ absorbance units ~ molecule expression in the strain.
         
         So, I think it's reasonable to proceed saying our method is fairly effective, with an
         accuracy +/- 500 units. Our next objective will then be to remove outliers, or note them,
         and FINALLY identify the prime candidate strains that are statistically significant
         improvements over the parent strain; these will move on to the next stage of
         selection. 
         
         ---
         
         Let's now, for the sake of closure, finality, look at all these populations now
         grouped together, split by well type rather than robot.
         """
)


fig6a = px.histogram(
    d_df,
    x="detrended_value",
    color="well_type",
    hover_name="well_type",
    hover_data="detrended_value",
    marginal="violin",  # in margin, visual of dist/data
    opacity=0.5,
    barmode="overlay",
    histfunc="count",
    histnorm="density",
    nbins=50,
)

fig6b = sns.displot(
    data=d_df,
    x="detrended_value",
    hue="well_type",
    kind="kde",
)


tab1, tab2 = st.tabs(["Plotly Histogram", "SNS Density Plot"])

with tab1:
    st.plotly_chart(fig6a)

with tab2:
    st.pyplot(fig6b)

st.write(
    """
         Yeah so, that's awesome. Can see 'em all in one place,
         the product of all our labor and attention. The violin
         plots above the histogram are fairly helpful to see the 
         distributions/spreads of data for the different well types/strains.
         
         So, let's finally get to the exciting, productive part!!
         One last step: remove, or note outliers. Maybe they could be wicked productive
         strains, don't wanna just discard 'em.
         
         # The Penultimate Step
         ---
         
         Removing outliers from parent and candidate strains. Basically, 
         remove the values +/- 3 STDEVs outside each population. BUT, note
         the +3 STDEV candidate strains just in case those ones are legit awesome strains.
         """
)

with st.echo():
    from scipy import stats
    import numpy as np

    # get just the parent strain data
    ps_detrended = d_df[d_df.well_type == "Parent Strain"]

    # find outliers in the data; outside zscore of 3, 3 stdev; note gets absolute value so anything +/-
    ps_outliers = ps_detrended[(np.abs(stats.zscore(ps_detrended.detrended_value)) > 3)]

    # redefine data to NOT have those outliers
    ps_detrended = ps_detrended[
        (np.abs(stats.zscore(ps_detrended.detrended_value)) < 3)
    ]

    # show outliers found and removed, just 3 of em lol
    ps_outliers

st.write(
    "Now do for the candidate strains. Code is slightly different \
    to keep track of the positive outliers."
)

with st.echo():
    # get just the CANDIDATE strain data
    X_detrended = d_df[d_df.well_type == "Standard Well"]

    # find outliers in the data;
    # THIS LINE and next get the negative and positive ID'd
    # outside zscore of 3, 3 stdev; note gets absolute value so anything +/-
    X_outliers_negative = X_detrended[(stats.zscore(X_detrended.detrended_value)) < -3]

    X_outliers_positive = X_detrended[(stats.zscore(X_detrended.detrended_value)) >= 3]

    # and finally, get the cleaned up X data, stuff within 3 stdev/z-score of 3
    X_detrended = X_detrended[(np.abs(stats.zscore(X_detrended.detrended_value)) < 3)]

st.write(
    "Amount of negative outliers: ",
    len(X_outliers_negative),
    "Amount of positive outliers: ",
    len(X_outliers_positive),
    "Remaining non-outlier samples: ",
    len(X_detrended),
    """
    # FINALLY. We can now identify the best candidate(s) (such as myself).
    ---
    
    One could do a t-test, sure; or more simply we could just set a 
    threshold at > 99%% of the parent strains' expression. 
    """,
)

with st.echo():
    # 99% cutoff:
    final_cutoff = ps_detrended.detrended_value.quantile(0.99)

    # get the candidates that have a value above this cutoff
    super_candidates = X_detrended[(X_detrended.detrended_value > final_cutoff)]

st.write(
    "Number of excellent candidates: ",
    len(super_candidates),
    "YAAAAAAAAAAAASSSSSSSSSSSSSSSSSSSSSSS!!!!!!!!!!!!!!!!",
)

st.image("memes/tuco2.jpg", width=300)

st.write(
    "Ok, let's plot these, plus the positive outliers, and see what it looks like."
)


super_candidates["sus"] = False
X_outliers_positive["sus"] = True
# all_candidates = super_candidates.append(X_outliers_positive)

all_candidates = pd.concat([super_candidates, X_outliers_positive])
# framesToMerge = [pc_data, ps_data, x_data]
# d_df = pd.concat(framesToMerge)

fig7 = px.histogram(
    super_candidates,
    x="detrended_value",
    color="sus",
    hover_name="sus",
    hover_data="detrended_value",
    marginal="violin",  # in margin, visual of dist/data
    opacity=0.5,
    barmode="overlay",
    histfunc="count",
    histnorm="density",
    nbins=50,
)

st.plotly_chart(fig7)

col1, col2 = st.columns(2)

with col1:
    st.write(
        "Excluding the outliers",
        super_candidates.detrended_value.describe(),
    )
with col2:
    st.write(
        "Including the outliers",
        all_candidates.detrended_value.describe(),
    )

st.write(
    """
    Alrighty then! So we see that the strains of interest start above around 600 greater
    than the mean of the parent strain*, and basically become less numerous. 
    The max is about 1600 above that mean, but may be an outlier.
    
    (*Mathematically, greater than the mean of the candidate strains, but the detrending renders this at 0 for both the candidate strains and parent strain, for better or worse given the line of best fit predicted for either dataset.)

    There's 133 candidates we have here. I'll leave it as is, one could slice this different ways depending on 
    inclusion of the potential outliers or not. But I'll output a 1D vector of these 3 dataframes below.

    So, basically, for 100/5280 candidates, we see ~ 800 improvement in absorbance ~ molecule expression. 
    Evolution in action. 
    
    Let's see the exact math so we can have some readout for our overall selection/evolution process:
    """,
)

f = df[df.well_type == "Parent Strain"]

st.write(
    "Original mean of parent strain, before detrending: ",
    f.value.mean(),
    "\n",
    "Mean of our results including outliers, representing a delta over the parent strain: ",
    all_candidates.detrended_value.mean(),
    "Percent improvement in these samples, including outliers: ",
    all_candidates.detrended_value.mean() / f.value.mean(),
    "Percent improvement without outliers: ",
    super_candidates.detrended_value.mean() / f.value.mean(),
    "So, these numbers suggest somewhere around 7-9%% improvement in these strains; in 99 or 133/5280 strains. Neat!",
    "Ok, and to get the data out, one would just come down and save one of these dfs to csv etc.",
)


with st.echo():
    all_candidate_strains_counter = all_candidates.counter.values
    # super candidates, doesn't include outliers
    super_candidate_strains_counter = super_candidates.counter.values
    # potential outlier candidates
    outlier_counters = X_outliers_positive.counter.values

    # Sample output
    st.write(outlier_counters)

st.write(
    """
         And... THAT'S FINALLY IT! We've identified ~133 candidates based on a cutoff
         defined as being superior to 99%% of parent strain samples. Our methodology has included:
         - Detrending by ordinary least squares, line of best fit, in lieu of time-series analysis or repeated-measure ANOVA, in order
         to obtain a greater statsitical power and compare as many samples as possible
         - Confirmation of null hypothesis to pool samples from different machines
         - Thresholding, effectively p value <= 0.01 samples to select for next run
         - Did this all in a pretty, interactive way using several libraries and memes.
         - Takes about a day to get this all together in Streamlit/Plotly, far superior and with
         very readable syntax etc. using this than my experience with Shiny, Dash... PowerBI may
         be close in production speed but is not as extensible of course.
         - Deployment and embedding into my main website is then trivial with Github/Streamlit/Huggingface.
         - 10/10 recommend
         """
)
