import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


# Display an image:
st.write("""
## An image:
""")
im = Image.open('zebra.jfif')
st.image(im)
# print(im.size, im.mode)

st.write("## Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame({
    'first col': [1, 2, 3, 4],
    'second col': [10, 20, 30, 40]
}))

st.write('## A line chart:')
chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

st.write('## A map:')
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)


st.write("""
## A download button of a short CSV file:
""")
text_contents = '''
... Col1, Col2
... 123, 456
... 789, 000
... '''
st.download_button(
    label='Download CSV', data=text_contents,
    file_name='file.csv', mime='text/csv'
)


st.write('## An input box (that we can later use its input for the .xlsx file with the users data!):')
title = st.text_input('Movie title', 'Life of Brian')
st.write('The current movie title is', title)

# See that every time you change the movie name, the new name is printed (if you delete the '#').
# print(title)


st.write('## A number input:')
number = st.number_input('Insert a number', step=1)
st.write('The current number is ', number)

# Once again, the varaible 'number' now holds the current content of the number input.



















