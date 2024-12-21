#!/usr/bin/env python
# coding: utf-8

# In[1]:


import import_ipynb
from dataPreparation import combined_df, kazakhstan_data


# In[2]:


from statsmodels.tsa.stattools import adfuller
import pandas as pd

numeric_df = combined_df.select_dtypes(include=['number'])

correlation_with_target = numeric_df.corr()["Score"].sort_values(ascending=False)
print(correlation_with_target)

threshold = 0.5
selected_features = correlation_with_target[correlation_with_target > threshold].index.tolist()

selected_features.remove("Score")
print("Выбранные признаки для модели:", selected_features)

X = combined_df[selected_features]
y = combined_df["Score"]


# In[3]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

print("Проверка стационарности на обучающей выборке:")
result_train = adfuller(y_train)
print("ADF Statistic:", result_train[0])
print("p-value:", result_train[1])

if result_train[1] <= 0.05:
    print("Данные обучающей выборки стационарны.")
else:
    print("Данные обучающей выборки нестационарны. Применим дифференцирование.")


# In[4]:


data = combined_df[['Year', 'Score']].copy()
data.columns = ['ds', 'y']
data['ds'] = pd.to_datetime(data['ds'], format='%Y')


# In[12]:


import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

df = combined_df[combined_df['Country'] == 'Kazakhstan'].copy()

if 'ds' not in df.columns:
    df['ds'] = pd.to_datetime(df['Year'], format='%Y')

df = df.sort_values('ds')
df.set_index('ds', inplace=True)

print("Первые строки DataFrame:")
print(df.head())
print("Последние строки DataFrame:")
print(df.tail())

features = ['GDP', 'Health', 'Family', 'Freedom']
target = 'Score'

print("Количество пропущенных значений:")
print(df[features + [target]].isna().sum())

df[features + [target]] = df[features + [target]].ffill()

print("Количество пропущенных значений после заполнения:")
print(df[features + [target]].isna().sum())

# Масштабирование данных
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features + [target]])

# Функция для создания последовательностей
def create_sequences(data, target_index, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length, :-1])  
        y.append(data[i+sequence_length, target_index]) 
    return np.array(X), np.array(y)

sequence_length = 3  # Количество прошлых временных шагов для прогноза
target_index = -1  # Последний столбец (Score)
X, y = create_sequences(scaled_data, target_index, sequence_length)

print(f"Форма X: {X.shape}") 
print(f"Форма y: {y.shape}") 

# Разделение на train и test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"\nФорма X_train: {X_train.shape}")
print(f"Форма X_test: {X_test.shape}")
print(f"Форма y_train: {y_train.shape}")
print(f"Форма y_test: {y_test.shape}")

if len(X_train.shape) != 3:
    raise ValueError(f"X_train должно иметь 3 измерения, но имеет {len(X_train.shape)}")

# Построение модели LSTM
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2]))) 
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(LSTM(32, activation='relu'))
model.add(Dense(1))  

model.compile(optimizer='adam', loss='mse')
model.summary()

# Обучение модели
history = model.fit(X_train, y_train, epochs=50, batch_size=8, 
                    validation_data=(X_test, y_test), verbose=1)

predictions = model.predict(X_test)
kazakhstan_df = combined_df[combined_df['Country'] == 'Kazakhstan'].copy()
print(f"Форма predictions: {predictions.shape}")

# X_test имеет форму (samples, sequence_length, features)
# X_test_last_features = X_test[:, -1, :-1]  # (samples, features)

# full_pred = np.hstack((X_test_last_features, predictions))  
# full_y = np.hstack((X_test_last_features, y_test.reshape(-1,1)))  

# predictions_rescaled = scaler.inverse_transform(full_pred)[:, -1]
# y_test_rescaled = scaler.inverse_transform(full_y)[:, -1]

# mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
# r2 = r2_score(y_test_rescaled, predictions_rescaled)
# print(f"\nLSTM - MSE: {mse}, R²: {r2}")

# test_dates = df.index[train_size + sequence_length : train_size + sequence_length + len(y_test_rescaled)]

# print(f"Количество дат для теста: {len(test_dates)}")
# print(f"Количество предсказаний: {len(predictions_rescaled)}")

# if len(test_dates) != len(predictions_rescaled):
#     raise ValueError("Длины 'test_dates' и 'predictions_rescaled' не совпадают!")

# plt.figure(figsize=(10, 6))
# plt.plot(df.index, df[target], label='Реальные значения', color='blue')
# plt.plot(test_dates, predictions_rescaled, label='Прогноз LSTM', color='green', linestyle='--')
# plt.title('Прогноз Happiness Score с использованием LSTM')
# plt.xlabel('Год')
# plt.ylabel('Happiness Score')
# plt.legend()
# plt.grid(True)
# plt.show()

X_test_last_features = X_test[:, -1, :-1]  

dummy_column = np.zeros((X_test_last_features.shape[0], 1)) 
full_pred = np.hstack((X_test_last_features, dummy_column, predictions))  
full_y = np.hstack((X_test_last_features, dummy_column, y_test.reshape(-1, 1))) 

# Обратное масштабирование
predictions_rescaled = scaler.inverse_transform(full_pred)[:, -1]
y_test_rescaled = scaler.inverse_transform(full_y)[:, -1]

# Оценка качества модели
mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
r2 = r2_score(y_test_rescaled, predictions_rescaled)
print(f"\nLSTM - MSE: {mse:.4f}, R²: {r2:.4f}")

test_dates = df.index[train_size + sequence_length : train_size + sequence_length + len(y_test_rescaled)]

plt.figure(figsize=(10, 6))
plt.plot(df.index, df[target], label='Реальные значения', color='blue')
plt.plot(test_dates, predictions_rescaled, label='Прогноз LSTM', color='green', linestyle='--')
plt.title('Прогноз Happiness Score с использованием LSTM')
plt.xlabel('Год')
plt.ylabel('Happiness Score')
plt.legend()
plt.grid(True)
plt.show()


# In[49]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

train_size = int(len(X) * 0.7)  
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Форма X_train: {X_train.shape}, Форма X_test: {X_test.shape}")
print(f"Форма y_train: {y_train.shape}, Форма y_test: {y_test.shape}")

# Обучение LSTM
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2]))) 
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(LSTM(32, activation='relu'))
model.add(Dense(1))  

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test), verbose=1)

predictions = model.predict(X_test)

# Обратное масштабирование
X_test_last_features = X_test[:, -1, :-1]  # Берем только признаки для последней выборки

# Добавляем фиктивный столбец для совместимости
dummy_column = np.zeros((X_test_last_features.shape[0], 1))

full_pred = np.hstack((X_test_last_features, dummy_column, predictions))
full_y = np.hstack((X_test_last_features, dummy_column, y_test.reshape(-1, 1)))

predictions_rescaled = scaler.inverse_transform(full_pred[:, :-1])[:, -1]  
y_test_rescaled = scaler.inverse_transform(full_y[:, :-1])[:, -1]  

mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
print(f"LSTM - MSE: {mse:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(df.index[-len(y_test_rescaled):], y_test_rescaled, label='Реальные значения', color='blue')
plt.plot(df.index[-len(predictions_rescaled):], predictions_rescaled, label='Прогноз LSTM', linestyle='--', color='green')
plt.title('Прогноз Happiness Score с использованием LSTM')
plt.xlabel('Год')
plt.ylabel('Happiness Score')
plt.legend()
plt.grid(True)
plt.show()

# Генерация будущих предсказаний
n_steps = 3  
last_input = X_test[0]  

future_preds = []
for _ in range(n_steps):
    next_pred = model.predict(last_input.reshape(1, X_train.shape[1], X_train.shape[2]))  # Предсказание
    future_preds.append(next_pred[0][0])
    
    last_input = np.roll(last_input, -1, axis=0)
    last_input[-1, -1] = next_pred[0][0]

last_date = pd.to_datetime(df.index[-1])  # Последняя дата в данных
future_dates = pd.date_range(last_date, periods=n_steps + 1, freq='YE')[1:]

# Обратное масштабирование будущих прогнозов
future_preds_rescaled = scaler.inverse_transform(
    np.hstack((np.tile(X_test[:, -1, :-1].mean(axis=0), (n_steps, 1)), 
               np.array(future_preds).reshape(-1, 1)))
)[:, -1]

plt.figure(figsize=(10, 6))
plt.plot(df.index[-len(y_test_rescaled):], y_test_rescaled, label='Реальные значения', color='blue')
plt.plot(future_dates, future_preds_rescaled, label='Прогноз LSTM', linestyle='--', color='green')
plt.title('Прогноз Happiness Score с использованием LSTM')
plt.xlabel('Год')
plt.ylabel('Happiness Score')
plt.legend()
plt.grid(True)
plt.show()




# In[35]:





# In[31]:


forecast_dates = pd.date_range(start=kazakhstan_df.index[-1] + pd.DateOffset(years=1), 
                               periods=n_future_steps, 
                               freq='YE')


# In[38]:


plt.figure(figsize=(12, 6))
plt.plot(kazakhstan_df.index, kazakhstan_df['Score'], label='Реальные значения', color='blue')
plt.plot(forecast_dates[:len(preds_rescaled)], preds_rescaled, 
         label='Прогноз LSTM', linestyle='--', color='green')  # Прогноз на 2019-2021
plt.plot(future_dates, future_preds_rescaled, label='Будущий прогноз', linestyle='--', color='red')
plt.title('Прогноз Happiness Score с использованием LSTM для Казахстана')
plt.xlabel('Год')
plt.ylabel('Happiness Score')
plt.legend()
plt.grid(True)
plt.show()


# In[9]:


forecast_dates = pd.date_range(start=kazakhstan_data.index[-1] + pd.DateOffset(years=1), 
                               periods=2, freq='YE') 

future_dates = pd.date_range(start=forecast_dates[-1] + pd.DateOffset(years=1), 
                             periods=3, freq='YE')  

plt.figure(figsize=(12, 6))

plt.plot(kazakhstan_data.index, kazakhstan_data['Score'], label='Реальные значения', color='blue')

plt.plot(forecast_dates, preds_rescaled[-2:], label='Прогноз LSTM', linestyle='--', color='green')

plt.plot(future_dates, future_preds_rescaled, label='Будущий прогноз', linestyle='--', color='red')

plt.title('Прогноз Happiness Score с использованием LSTM для Казахстана')
plt.xlabel('Год')
plt.ylabel('Happiness Score')
plt.legend()
plt.grid(True)
plt.show()


# In[28]:


forecast_dates = pd.date_range(start=kazakhstan_df.index[-1] + pd.DateOffset(years=1), 
                               periods=5, freq='YE') 

future_dates = pd.date_range(start=forecast_dates[-1] + pd.DateOffset(years=1), 
                             periods=3, freq='YE')  

plt.figure(figsize=(12, 6))

plt.plot(kazakhstan_df.index, kazakhstan_df['Score'], label='Реальные значения', color='blue')

plt.plot(forecast_dates, preds_rescaled[-5:], label='Прогноз LSTM', linestyle='--', color='green')

plt.title('Прогноз Happiness Score с использованием LSTM для Казахстана')
plt.xlabel('Год')
plt.ylabel('Happiness Score')
plt.legend()
plt.grid(True)
plt.show()


# In[10]:


combined_df.columns


# In[11]:


combined_df.head()


# In[13]:


kazakhstan_df = combined_df[combined_df['Country'] == 'Kazakhstan'].copy()


# In[15]:


print(combined_df['Country'].unique())


# In[16]:


kazakhstan_df


# In[17]:


print(kazakhstan_df.index)
print(type(kazakhstan_df.index[0]))


# In[19]:


kazakhstan_df['Year'] = pd.to_datetime(kazakhstan_df['Year'], format='%Y')
kazakhstan_df.set_index('Year', inplace=True)


# In[25]:


scaler.fit(combined_df[['GDP', 'Family', 'Health', 'Freedom', 'Score']])


# In[26]:


kazakhstan_scaled = scaler.transform(kazakhstan_df[['GDP', 'Family', 'Health', 'Freedom', 'Score']])


# In[27]:


# Создание последовательностей
X_kazakhstan, y_kazakhstan = create_sequences(kazakhstan_scaled, target_index, sequence_length)

# Прогноз
preds = model.predict(X_kazakhstan)

# Обратное масштабирование
dummy_column = np.zeros((preds.shape[0], 1))
full_input = np.hstack((X_kazakhstan[:, -1, :-1], dummy_column, preds.reshape(-1, 1)))
preds_rescaled = scaler.inverse_transform(full_input)[:, -1]


# In[30]:


print(X_kazakhstan.shape)
print(model.input_shape)  


# In[32]:


print(X_train.shape)  


# In[50]:


plt.figure(figsize=(12, 6))

plt.plot(kazakhstan_df.index, kazakhstan_df['Score'], label='Реальные значения', color='blue')

connected_dates = np.concatenate([kazakhstan_df.index[-sequence_length:], forecast_dates])
connected_values = np.concatenate([kazakhstan_df['Score'].values[-sequence_length:], preds_rescaled])
plt.plot(connected_dates, connected_values, label='Прогноз LSTM', linestyle='--', color='green')

plt.plot(future_dates, future_preds_rescaled, label='Будущий прогноз', linestyle='--', color='red')

plt.title('Прогноз Happiness Score с использованием LSTM для Казахстана')
plt.xlabel('Год')
plt.ylabel('Happiness Score')
plt.legend()
plt.grid(True)
plt.show()


# In[54]:


plt.figure(figsize=(12, 6))

plt.plot(kazakhstan_df.index, kazakhstan_df['Score'], label='Реальные значения', color='blue')

last_real_date = kazakhstan_df.index[-1] 
first_forecast_date = forecast_dates[0] 

smooth_dates = pd.date_range(start=last_real_date, end=first_forecast_date, periods=10)

connected_dates = smooth_dates.append(forecast_dates)

smooth_values = np.linspace(kazakhstan_df['Score'].values[-1], preds_rescaled[0], num=10)

# Объединяем значения: реальные + прогноз
connected_values = np.concatenate([smooth_values, preds_rescaled])

plt.plot(
    connected_dates,
    connected_values,
    label='Сглаженный прогноз LSTM',
    linestyle='--',
    color='green'
)

plt.plot(future_dates, future_preds_rescaled, label='Будущий прогноз', linestyle='--', color='red')

plt.title('Прогноз Happiness Score с использованием LSTM для Казахстана')
plt.xlabel('Год')
plt.ylabel('Happiness Score')
plt.legend()
plt.grid(True)
plt.show()


# In[45]:


print(f"X_kazakhstan shape: {X_kazakhstan.shape}")  
print(f"Model input shape: {model.input_shape}")


# In[46]:


X_kazakhstan = np.squeeze(X_kazakhstan, axis=-1) 


# In[57]:


plt.figure(figsize=(12, 6))

plt.plot(kazakhstan_df.index, kazakhstan_df['Score'], label='Реальные значения', color='blue')

last_real_date = kazakhstan_df.index[-1]

smooth_dates = pd.date_range(start=last_real_date, end=forecast_dates[0], periods=5)

connected_dates = pd.Series(np.concatenate([smooth_dates, forecast_dates])).unique()

smooth_values = np.linspace(kazakhstan_df['Score'].values[-1], preds_rescaled[0], num=len(smooth_dates))

connected_values = np.concatenate([smooth_values, preds_rescaled[:len(connected_dates) - len(smooth_values)]])

assert len(connected_dates) == len(connected_values), "Несоответствие длин дат и значений!"

plt.plot(
    connected_dates,
    connected_values,
    label='Сглаженный прогноз LSTM',
    linestyle='--',
    color='green'
)

plt.plot(future_dates, future_preds_rescaled, label='Будущий прогноз', linestyle='--', color='red')

plt.title('Прогноз Happiness Score с использованием LSTM для Казахстана')
plt.xlabel('Год')
plt.ylabel('Happiness Score')
plt.legend()
plt.grid(True)
plt.show()


# In[60]:


plt.figure(figsize=(12, 6))

plt.plot(kazakhstan_df.index, kazakhstan_df['Score'], label='Реальные значения', color='blue')

last_real_date = kazakhstan_df.index[-1]
last_real_value = kazakhstan_df['Score'].values[-1]

# Промежуточные даты для сглаживания (между последней реальной и первой прогнозной)
smooth_dates = pd.date_range(start=last_real_date, end=forecast_dates[0], periods=5)[1:]

smooth_values = np.linspace(last_real_value, preds_rescaled[0], num=len(smooth_dates))

connected_dates = np.concatenate([kazakhstan_df.index[-1:], smooth_dates, forecast_dates])
connected_values = np.concatenate([[last_real_value], smooth_values, preds_rescaled])

assert len(connected_dates) == len(connected_values), "Несоответствие длин дат и значений!"

plt.plot(
    connected_dates,
    connected_values,
    label='Сглаженный прогноз LSTM',
    linestyle='--',
    color='green'
)

plt.plot(future_dates, future_preds_rescaled, label='Будущий прогноз', linestyle='--', color='red')

plt.title('Прогноз Happiness Score с использованием LSTM для Казахстана')
plt.xlabel('Год')
plt.ylabel('Happiness Score')
plt.legend()
plt.grid(True)
plt.show()


# In[61]:


plt.figure(figsize=(12, 6))

plt.plot(kazakhstan_df.index, kazakhstan_df['Score'], label='Реальные значения', color='blue')

last_real_date = kazakhstan_df.index[-1]
last_real_value = kazakhstan_df['Score'].values[-1]

smooth_dates = pd.date_range(start=last_real_date, end=forecast_dates[0], periods=5)

smooth_values = np.linspace(last_real_value, preds_rescaled[0], num=len(smooth_dates))

connected_dates = np.concatenate([smooth_dates, forecast_dates])
connected_values = np.concatenate([smooth_values, preds_rescaled])

assert len(connected_dates) == len(connected_values), "Несоответствие длин дат и значений!"

plt.plot(
    connected_dates,
    connected_values,
    label='Сглаженный прогноз LSTM',
    linestyle='--',
    color='green'
)

plt.plot(future_dates, future_preds_rescaled, label='Будущий прогноз', linestyle='--', color='red')

plt.title('Прогноз Happiness Score с использованием LSTM для Казахстана')
plt.xlabel('Год')
plt.ylabel('Happiness Score')
plt.legend()
plt.grid(True)
plt.show()


# In[62]:


pd.date_range


# In[63]:


forecast_dates


# In[64]:


preds_rescaled


# In[65]:


kazakhstan_df


# In[66]:


future_dates 


# In[67]:


future_preds_rescaled


# In[69]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# future_dates = pd.to_datetime(['2024-12-31', '2025-12-31', '2026-12-31'])
# future_preds_rescaled = np.array([6.21237382, 6.17682541, 6.13521123])

last_real_date = kazakhstan_df.index[-1]

forecast_dates = future_dates
preds_rescaled = future_preds_rescaled

smooth_dates = pd.date_range(start=last_real_date, end=forecast_dates[0], periods=6)[1:]

smooth_values = np.linspace(kazakhstan_df['Score'].values[-1], preds_rescaled[0], num=len(smooth_dates))

connected_dates = np.concatenate((kazakhstan_df.index, smooth_dates, forecast_dates))
connected_values = np.concatenate((kazakhstan_df['Score'].values, smooth_values, preds_rescaled))

assert len(connected_dates) == len(connected_values), "Длины дат и значений не совпадают!"

plt.figure(figsize=(12, 6))

plt.plot(kazakhstan_df.index, kazakhstan_df['Score'], label='Реальные значения', color='blue')

plt.plot(
    connected_dates,
    connected_values,
    label='Сглаженный прогноз LSTM',
    linestyle='--',
    color='green'
)

plt.plot(forecast_dates, preds_rescaled, label='Будущий прогноз', linestyle='--', color='red')

plt.title('Прогноз Happiness Score с использованием LSTM для Казахстана')
plt.xlabel('Год')
plt.ylabel('Happiness Score')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




