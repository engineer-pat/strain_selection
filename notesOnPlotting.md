https://docs.streamlit.io/library/api-reference/charts
For future me:
Streamlit has multiple options for charts. If you want
something static, it has good basic graphs. (By static, I mean
not interactable etc. can probably still do live dashboard.)
Otherwise they piggyback off everyone else, including for the basic graphs lol they're just wrapped.
So, ultimately you can do:
Interaction, Plotly:
https://docs.streamlit.io/library/api-reference/charts/st.plotly_chart
THIS LOOKS ABSOLUTELY FIRE FOR DASHBOARDS:
https://github.com/okld/streamlit-elements

Good, detailed, A CORUNCOPIA, static:
Vega-Altair: https://altair-viz.github.io/gallery/
... And plost as a wrapper, make it even easier to use em:
https://plost.streamlit.app/
But still static. 

Bokeh seems similarly good for niche/common stuff:
https://bokeh.org/

Pydeck seems dope for anything map/population wise:
https://docs.streamlit.io/library/api-reference/charts/st.pydeck_chart

ALSO FUTURE ME FOR GITHUB MIRRORING, USING GITHUB ACTIONS:
https://huggingface.co/docs/hub/spaces-github-actions

*
use git push space
named the remote/origin 'space' for HF.