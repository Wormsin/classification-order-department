import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import numpy as np
import dash_bootstrap_components as dbc
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
from transformers import  AutoModelForSequenceClassification
from transformers import AutoTokenizer

# Ваша модель (подставьте реальную модель здесь)
model = AutoModelForSequenceClassification.from_pretrained("bert_model_3")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
# Функция для расчета энтропии предсказания
def calculate_entropy(probabilities):
    return -np.sum(probabilities * np.log2(probabilities))

file_name = "cleaned_data.csv"
df = pd.read_csv(file_name, sep = '\t')
# Названия отделов
departments = np.unique(df['Отдел'].values)


def prepare_data_for_bert(text, tokenizer, max_length=128):
    # Токенизация и преобразование текста
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,        # Добавляем специальные токены [CLS] и [SEP]
        max_length=max_length,          # Ограничиваем длину текста
        padding='max_length',           # Добавляем паддинг до max_length
        truncation=True,                # Обрезаем текст до max_length, если нужно
        return_tensors='pt',            # Возвращаем тензоры PyTorch
    )
    
    # Возвращаем идентификаторы токенов, маску внимания
    return encoding['input_ids'], encoding['attention_mask']


# Создание веб-приложения Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Предсказание отдела и энтропия")),
    ], className="my-4"),
    
    dbc.Row([
        dbc.Col(dbc.Input(id="input-data", type="text", placeholder="Введите данные для модели", style={'width': '100%'})),
        dbc.Col(dbc.Button("Предсказать", id="predict-button", color="primary", className="ml-2")),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col(html.Div(id="output-result", style={'marginTop': 20})),
    ]),
])

# Колбэк для обработки данных и отображения результата
@app.callback(
    Output("output-result", "children"),
    Input("predict-button", "n_clicks"),
    State("input-data", "value"),
)


def predict_department(n_clicks, input_data):
    if n_clicks is None or not input_data:
        return "Введите данные и нажмите 'Предсказать'"
    
    input_ids, attention_mask = prepare_data_for_bert(input_data, tokenizer)
  
 
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1).numpy()[0]
    
    entropy = calculate_entropy(np.array(probabilities))
    
    # Форматирование результатов для вывода
    #department_probs = {dep: np.round(prob*100, decimals=2) for dep, prob in zip(departments, probabilities)}
    department_probs = [
        html.Li(
            f"{dep}: {round(prob*100, 2)}",
            style={'color': 'red'} if prob > 0.5 else {'color': 'black'}
        ) for dep, prob in zip(departments, probabilities)
    ]
    
    
    result_text = [
        html.H4("Результаты предсказания:"),
        html.Ul(department_probs),
        html.P(f"Энтропия предсказания: {float(np.round(entropy*100)/100)}"),
    ]
    return result_text

if __name__ == "__main__":
    app.run_server(debug=True, port = 8051)
