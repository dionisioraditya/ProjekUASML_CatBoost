import streamlit as st
import pickle
import pandas as pd

with open('BestModel_RandomForest_CatBoost.pkl', 'rb') as f:
    model = pickle.load(f)

with open('columns.pkl', 'rb') as f:
    training_columns = pickle.load(f)

st.title('🚗 Prediksi Harga Mobil (RandomForest)')
st.write('Masukkan spesifikasi mobil untuk memprediksi harganya')

st.divider()

col1, col2 = st.columns(2)

with col1:
    levy = st.number_input('Levy', min_value=0, value=500)
    prod_year = st.number_input('Tahun Produksi', min_value=1990, max_value=2025, value=2015)
    engine_volume = st.number_input('Engine Volume (L)', min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    mileage = st.number_input('Mileage (km)', min_value=0, value=100000)
    cylinders = st.selectbox('Cylinders', [1, 2, 3, 4, 5, 6, 8, 10, 12], index=3)

with col2:
    airbags = st.slider('Airbags', min_value=0, max_value=16, value=4)
    leather_interior = st.selectbox('Leather Interior', ['Yes', 'No'])
    fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'Hybrid', 'LPG', 'CNG', 'Plug-in Hybrid'])
    gear_box = st.selectbox('Gear Box', ['Automatic', 'Manual', 'Tiptronic', 'Variator'])
    drive_wheels = st.selectbox('Drive Wheels', ['Front', 'Rear', '4x4'])

col3, col4 = st.columns(2)

with col3:
    manufacturer = st.selectbox('Manufacturer', [
        'TOYOTA', 'HONDA', 'HYUNDAI', 'MERCEDES-BENZ', 'BMW', 'LEXUS', 
        'CHEVROLET', 'FORD', 'NISSAN', 'KIA', 'VOLKSWAGEN', 'AUDI',
        'SUBARU', 'MAZDA', 'MITSUBISHI', 'JEEP', 'PORSCHE', 'LAND ROVER',
        'BUICK', 'CADILLAC', 'GMC', 'ACURA', 'INFINITI', 'VOLVO',
        'JAGUAR', 'ALFA ROMEO', 'FIAT', 'MINI', 'PEUGEOT', 'RENAULT',
        'SKODA', 'SUZUKI', 'DAEWOO', 'OPEL', 'SSANGYONG', 'VAZ', 'GAZ',
        'ZAZ', 'CHRYSLER', 'DODGE', 'HUMMER', 'SCION', 'SATURN', 'LINCOLN',
        'MERCURY', 'PONTIAC', 'SAAB', 'LANCIA', 'ROVER', 'SEAT', 'CITROEN',
        'DAIHATSU', 'ISUZU', 'GREATWALL', 'HAVAL', 'TESLA', 'BENTLEY',
        'FERRARI', 'LAMBORGHINI', 'MASERATI', 'ROLLS-ROYCE', 'ASTON MARTIN'
    ])
    category = st.selectbox('Category', [
        'Sedan', 'Jeep', 'Hatchback', 'Coupe', 'Universal', 'Minivan',
        'Goods wagon', 'Microbus', 'Pickup', 'Cabriolet', 'Limousine'
    ])

with col4:
    wheel = st.selectbox('Wheel', ['Left wheel', 'Right-hand drive'])
    doors = st.selectbox('Doors', ['2-3', '4-5'])
    color = st.selectbox('Color', [
        'Black', 'White', 'Silver', 'Grey', 'Blue', 'Red', 'Green',
        'Brown', 'Beige', 'Golden', 'Orange', 'Yellow', 'Purple',
        'Pink', 'Carnelian red', 'Sky blue'
    ])
    model_car = st.text_input('Model Mobil', value='Camry')

st.divider()

if st.button('🔮 Prediksi Harga', use_container_width=True):
    
    input_df = pd.DataFrame(columns=training_columns)
    input_df.loc[0] = 0
    
    input_df['Levy'] = levy
    input_df['Prod. year'] = prod_year
    input_df['Leather interior'] = 1 if leather_interior == 'Yes' else 0
    input_df['Engine volume'] = engine_volume
    input_df['Mileage'] = mileage
    input_df['Cylinders'] = cylinders
    input_df['Doors'] = 2 if doors == '2-3' else 4
    input_df['Airbags'] = airbags
    
    one_hot_mapping = {
        f'Manufacturer_{manufacturer}': 1,
        f'Category_{category}': 1,
        f'Fuel type_{fuel_type}': 1,
        f'Gear box type_{gear_box}': 1,
        f'Drive wheels_{drive_wheels}': 1,
        f'Wheel_{wheel}': 1,
        f'Color_{color}': 1,
        f'Model_{model_car}': 1
    }
    
    for col, val in one_hot_mapping.items():
        if col in input_df.columns:
            input_df[col] = val
    
    try:
        prediction = model.predict(input_df)
        st.success(f' 💰 Estimasi Harga: ${prediction[0]:,.2f}')
        
    except Exception as e:
        st.error(f'Error saat prediksi: {e}')