#!/usr/bin/env python
# coding: utf-8
# ----------------------------------------------------------------------------
# Created By  : Nicolas Ramosaj
# Created Date: 22/07/2024
# version ='1.0'
# ---------------------------------------------------------------------------
""" Streamlit of Swiss Lotto Analyser """

import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats as stats
import streamlit as st

from random import randint
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options

# ---------------------------------------------------------------------------


@st.experimental_singleton
def installff():
  os.system('sbase install geckodriver')
  os.system('ln -s /home/appuser/venv/lib/python3.7/site-packages/seleniumbase/drivers/geckodriver /home/appuser/venv/bin/geckodriver')

try:
    _ = installff()
except:
    print("Installation of Selenium does not work")

# ---------------------------------------------------------------------------



total_number = 42
total_lucky = 6


@st.cache_data()
def load_data() -> pd.DataFrame:
    """
    Return dataset with the numbers without complementary or powerball
    :return: pd.DataFrame, numbers drawn by date
    """
    cols = ['Date', 'No1', 'No2', 'No3', 'No4', 'No5', 'No6', 'Lucky']
    return pd.read_csv(filepath_or_buffer='data/swisslotto_2013_2024.csv', index_col=[0],
                       parse_dates=True, dayfirst=True, usecols=cols).sort_index()


def get_all_draws_from_last_draw(date: pd.DatetimeIndex) -> list:
    """
    Get the list of all missing draws
    :param date: pd.DatetimeIndex, last datetime from the dataset
    :return: list, list of all missing draws
    """
    return [d.strftime('%Y-%m-%d') for d in pd.date_range(start=date.strftime('%Y-%m-%d'),
                                                          end=pd.Timestamp.today().strftime('%Y-%m-%d'))
            if d.weekday() in [2, 5]]


@st.cache_data()
def scrapping_numbers(history: int) -> pd.DataFrame:
    """
    Scrap the missing draws from initial dataset
    :param history: int, length of missing draws
    :return: pd.DataFrame, updated dataset with last draws
    """
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Firefox(options=options)
    driver.get("https://www.swisslos.ch/fr/swisslotto/informations/r%C3%A9sultats/r%C3%A9sultats-gains.html")
    title = ['No1', 'No2', 'No3', 'No4', 'No5', 'No6', 'Lucky', 'Replay']

    data = {}
    for _ in range(0, history):
        date = (driver.find_element(by=By.XPATH, value='.//div[@class = "filter"]/form/label/div/input[1]')
                .get_attribute('value'))
        numbers = driver.find_elements(by=By.XPATH, value='.//div[@class = "quotes__game"]/div/ul/li/span')
        data[date] = {key: int(value.text) for key, value in zip(title, numbers)}
        driver.find_element(by=By.CLASS_NAME, value='filter-prev-draw.button__secondary').click()

    scrapped_df = pd.DataFrame(data=data).T.drop(columns=['Replay'])
    scrapped_df.index = pd.to_datetime(scrapped_df.index, format='%d.%m.%Y')
    return pd.concat([st.session_state.data_lotto.tail(), scrapped_df]).sort_index()


def count_unique(df: pd.DataFrame) -> np.ndarray:
    """
    Count unique number through the dataset
    :param df: pd.DataFrame, dataset of draws
    :return: np.ndarray, array of unique
    """
    _, count = np.unique(df.values, return_counts=True)

    return count


def count_frequency_poisson(df: pd.DataFrame) -> tuple:
    """
    Count the poisson distribution of gap between drawn
    :param df: pd.DataFrame, dataset of draws
    :return: tuple (np.ndarray, np.ndarray), poisson distribution by number, gap from last draw
    """
    df_freq = df.reset_index().drop(columns='Date')
    # Dictionary of draw index where the number occurs
    arg_by_number = {num: [] for num in range(1, total_number + 1)}
    for row in df_freq.iterrows():
        for col in df_freq.columns:
            arg_by_number[int(row[1][col])].append(row[0])

    # Dictionary of gap where the number occurs
    draw_gap_by_number = {key: np.diff(values) for key, values in arg_by_number.items()}

    # Compute poisson distribution by number
    x_poisson = np.arange(start=0, stop=50)
    poisson_by_number = {num: [] for num in range(1, total_number + 1)}
    for key, values in draw_gap_by_number.items():
        # Compute poisson distribution
        poisson_by_number[key].append([stats.poisson.pmf(k=x, mu=np.mean(values)) for x in x_poisson])

    # Probability to occur at next draw
    prob_to_occur = {}
    drawn_gap = {}
    for key, values in arg_by_number.items():
        last_occur = np.max(values).astype(int)
        drawn_gap[key] = df_freq.index[-1] - last_occur
        prob_to_occur[key] = np.round(poisson_by_number[key][0][drawn_gap[key]], decimals=5)

    return np.array(list(prob_to_occur.values())), np.array(list(drawn_gap.values()))


def generate_draw() -> dict:
    """
    Generate draws
    :return: dict, draws with their probabilities
    """
    data = {
        i: {'numbers': sorted([randint(1, total_number), randint(1, total_number),
                               randint(1, total_number), randint(1, total_number),
                               randint(1, total_number), randint(1, total_number)]),
            'Lucky': randint(1, total_lucky),
            'probability_1': 1.0,
            'probability_freq_1': 1.0}
        for i in range(0, 20)
    }

    return data


def check_and_sort_draws(data: dict, prob_by_unique: np.ndarray, prob_by_freq: np.ndarray) -> pd.DataFrame:
    """
    Check and sort generated draws
    :param data: dict, generated draw
    :param prob_by_unique: np.ndarray, count by unique
    :param prob_by_freq: np.ndarray, count by frequency
    :return: pd.DataFrame, data checked
    """
    for key, values in data.items():
        if np.unique(values['numbers']).shape[0] < 6:
            values['probability_1'] *= 0.0
            values['probability_freq_1'] *= 0.0
        else:
            for num in values['numbers']:
                values['probability_1'] *= prob_by_unique[num - 1]
                values['probability_freq_1'] *= prob_by_freq[num - 1]

    df_draw = pd.DataFrame(data).T
    df_draw[['No1', 'No2', 'No3', 'No4', 'No5', 'No6']] = pd.DataFrame(df_draw.numbers.tolist(),
                                                                       index=df_draw.index)
    df_draw = df_draw.drop(columns='numbers').sort_values(by=['probability_freq_1'], ascending=False,
                                                          ignore_index=True)
    df_draw = df_draw[['No1', 'No2', 'No3', 'No4', 'No5', 'No6', 'Lucky', 'probability_1', 'probability_freq_1']]
    return df_draw.head(1)


def get_lucky_numbers(draw: pd.DataFrame, prob_by_freq: np.ndarray):
    """
    Get most probable numbers
    :param draw: pd.DataFrame, generated draw
    :param prob_by_freq: np.ndarray, count by frequency
    :return: pd.DataFrame, data checked
    """
    my_numbers = {num[1].values[0]: prob_by_freq[int(num[1].values[0])-1]
                  for num in draw.drop(columns=['Lucky']).T.iterrows()}
    my_favorite = list(dict(sorted(my_numbers.items(), key=lambda x: x[1], reverse=True)).keys())[:2]
    return pd.DataFrame({"numbers": [my_favorite]})


def plot_graphic(data: np.ndarray, draw: pd.DataFrame, y_title="Count") -> object:
    """
    Plot the graphics summarizing the draw
    :param data: np.ndarray, count of something
    :param draw: pd.DataFrame, generated draw
    :param y_title: str, y_axis name
    :return: object, figure to plot
    """
    default_color = "blue"
    colors = {num: "red" for num in draw.drop(columns=['Lucky']).values.flatten().tolist()}
    numbers = np.arange(start=1, stop=43, step=1)

    color_discrete_map = [
        colors.get(i, default_color)
        for i in numbers]

    figure_bar = px.bar(x=numbers, y=data, color=color_discrete_map, color_discrete_map={"blue": "#17becf",
                                                                                         "red": "#d62728"})
    figure_bar.update_traces(showlegend=False)
    figure_bar.update_layout(xaxis_title="Nombre", yaxis_title=y_title)

    return figure_bar


# --- Main Program -----------------------------------------------------------

if 'data_lotto' not in st.session_state:
    st.session_state.data_lotto = load_data()
    missing_draws = get_all_draws_from_last_draw(date=st.session_state.data_lotto.index[-1] + pd.DateOffset(1))
    if len(missing_draws) and pd.Timestamp.today().hour > 21:
        st.session_state.data_lotto = scrapping_numbers(history=len(missing_draws))

st.title("Swiss Lotto Analyser")

if 'my_draw' not in st.session_state:
    st.session_state.prob_by_unique = count_unique(df=st.session_state.data_lotto.drop(columns=['Lucky']))
    st.session_state.prob_by_freq, st.session_state.drawn_gap = count_frequency_poisson(df=st.session_state.data_lotto)
    st.session_state.my_draw = check_and_sort_draws(data=generate_draw(),
                                                    prob_by_unique=st.session_state.prob_by_unique,
                                                    prob_by_freq=st.session_state.prob_by_freq).drop(
        columns=['probability_1', 'probability_freq_1'])

st.subheader("Ceci est votre tirage chance")
st.markdown("Ce sont les 6 numéros ainsi que 1 numéro chance. La combinaison choisies se base sur "
            "sur la meilleure estimation de 100 tirages aléatoires.")
st.dataframe(data=st.session_state.my_draw, use_container_width=True, hide_index=True)

lucky = get_lucky_numbers(draw=st.session_state.my_draw, prob_by_freq=st.session_state.prob_by_freq)
st.data_editor(lucky, column_config={"numbers": st.column_config.ListColumn(label="Vos numéros porte-bonheur",
                                                                            help="Ce sont vos numéros porte-bonheur",
                                                                            width="medium")}, hide_index=True)

if st.button("Nouveau tirage"):
    st.session_state.my_draw = check_and_sort_draws(data=generate_draw(),
                                                    prob_by_unique=st.session_state.prob_by_unique,
                                                    prob_by_freq=st.session_state.prob_by_freq).drop(
        columns=['probability_1', 'probability_freq_1'])
    st.experimental_rerun()

st.subheader("Le nombre de tirage de chaque numéro")
st.markdown("Ce graphique montre le nombre de fois que chaque numéro a été tiré depuis 2013. "
            "En rouge, ce sont vos numéros.")
figure_unique = plot_graphic(data=st.session_state.prob_by_unique, draw=st.session_state.my_draw, y_title="Compteur")
st.plotly_chart(figure_or_data=figure_unique, use_container_width=True)

st.subheader("Le nombre de tirage depuis la dernière apparition")
st.markdown("Ce graphique montre le nombre de tirages depuis la dernière apparition du numéro. "
            "En rouge, ce sont vos numéros.")
figure_gap = plot_graphic(data=st.session_state.drawn_gap, draw=st.session_state.my_draw, y_title="Compteur")
st.plotly_chart(figure_or_data=figure_gap, use_container_width=True)

st.subheader("La probabilté de sortir au prochain tirage")
st.markdown("Ce graphique montre la probabilité de chaque numéro de sortir au prochain tirage (basé sur une "
            "distribution de poisson). En rouge, ce sont vos numéros.")
figure_freq = plot_graphic(data=st.session_state.prob_by_freq, draw=st.session_state.my_draw, y_title="Probabilité (%)")
st.plotly_chart(figure_or_data=figure_freq, use_container_width=True)
