model_lstm_opt1 = Sequential([
    LSTM(50, return_sequences=True, input_shape=(64, 64)),
    LSTM(50),
    Dense(1, activation='sigmoid')
])

model_lstm_opt1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_lstm1 = model_lstm_opt1.fit(X_train_lstm, y_train, epochs=10, validation_split=0.2, batch_size=32)

#Performans değerlendirmesi
plt.plot(history_lstm1.history['loss'], label='Training loss')
plt.plot(history_lstm1.history['val_loss'], label='Validation loss')
plt.title('Training and Validation Loss')

plt.plot(history_lstm1.history['accuracy'], label='Training accuracy')
plt.plot(history_lstm1.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()



model_lstm_opt2 = Sequential([
    LSTM(50, return_sequences=True, input_shape=(64, 64)),
    Dropout(0.5),
    LSTM(50),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model_lstm_opt2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_lstm2 = model_lstm_opt2.fit(X_train_lstm, y_train, epochs=10, validation_split=0.2, batch_size=32)

plt.plot(history_lstm2.history['loss'], label='Training loss')
plt.plot(history_lstm2.history['val_loss'], label='Validation loss')
plt.title('Training and Validation Loss')

plt.plot(history_lstm2.history['accuracy'], label='Training accuracy')
plt.plot(history_lstm2.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model_lstm_opt3 = Sequential([
    LSTM(50, input_shape=(64, 64)),
    Dense(1, activation='sigmoid')
])

model_lstm_opt3.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

history_lstm3 = model_lstm_opt3.fit(X_train_lstm, y_train, epochs=10, validation_split=0.2, batch_size=32)

plt.plot(history_lstm3.history['loss'], label='Training loss')
plt.plot(history_lstm3.history['val_loss'], label='Validation loss')
plt.title('Training and Validation Loss')

plt.plot(history_lstm3.history['accuracy'], label='Training accuracy')
plt.plot(history_lstm3.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
