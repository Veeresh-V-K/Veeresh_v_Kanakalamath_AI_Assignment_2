# Voice Cloning Model

This repository contains a voice cloning model trained on the VoxCeleb-1 dataset. The model is built using TensorFlow and utilizes the Long Short-Term Memory (LSTM) architecture.

## Requirements

- Python 3.x
- TensorFlow
- Librosa
- NumPy
- SoundFile
- PyDub
- gTTS
- Google Colab (optional)

## Dataset

The model is trained on the VoxCeleb-1 dataset, which is not included in this repository. To train the model, you need to obtain the dataset separately. Follow the steps below to set up the dataset:

1. Download the VoxCeleb-1 dataset from [source_link](https://example.com/source_link).
2. Place the downloaded dataset ZIP file in your Google Drive. Make sure you have enough storage space available.

## Setup

1. Mount Google Drive: Run the following command to mount your Google Drive in Google Colab:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Set the path to the VoxCeleb-1 dataset ZIP file: Update the `zip_file_path` variable in the code with the correct path to the dataset ZIP file in your Google Drive.

3. Extract the dataset ZIP file: The code will extract the dataset ZIP file automatically using the following code snippet:

   ```python
   with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
       zip_ref.extractall('voxceleb-1-dataset')
   ```

4. Preprocess audio files: The code will preprocess the audio files in the dataset, including normalization, padding, and reshaping. It will store the preprocessed audio files in the `audio_files` list.

5. Train the model: The preprocessed audio files will be used to train the voice cloning model. The model architecture consists of an input layer, a Reshape layer for compatibility with LSTM, an LSTM layer, and a Dense layer. The model is trained using the mean squared error (MSE) loss and the Adam optimizer. The number of epochs can be adjusted as needed.

   ```python
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(8000,)),
       tf.keras.layers.Reshape((8000, 1)),
       tf.keras.layers.LSTM(128),
       tf.keras.layers.Dense(1, activation='linear')
   ])
   model.compile(loss='mse', optimizer='adam')
   model.fit(audio_data, target_data, epochs=5)
   ```

6. Clean up: The extracted files from the dataset will be removed using the following code snippet:

   ```python
   shutil.rmtree('voxceleb-1-dataset')
   ```

## Generating Speech from Text

You can use the voice cloning model to generate speech from text. The following steps explain how to generate speech:

1. Generate text-to-speech audio: The code utilizes the gTTS (Google Text-to-Speech) library to generate speech from the given text. Update the `text` variable in the code with your desired text.

   ```python
   text = 'This is a test of the voice cloning model.'
   tts = gTTS(text=text, lang='en')
   tts.save('/content/generated_speech.mp3')
   ```

2. Preprocess the generated speech audio: The generated speech audio will be preprocessed similarly to the training data, including normalization, padding, and reshaping.

3. Generate audio using the model: The preprocessed speech audio will be passed through the voice cloning model to generate audio output.

   ```python
   generated_audio = model.predict(speech)
   ```

4. Post-process the generated audio: The generated audio will undergo post-processing steps, including power transformation, amplitude conversion, and resampling.

5. Save the generated speech audio: The final generated speech audio will be saved in both WAV and MP3 formats.

   ```python
   sf.write('/content/generated_speech.wav', generated_audio, 16000, format='WAV')
   audio_segment.export('/content/generated_speech_output.mp3', format='mp3')
   ```

## Evaluation

To evaluate the performance of the voice cloning model, you can consider the following metrics:

- Mean Squared Error (MSE): Measure the average squared difference between the generated audio and the target audio.

- Signal-to-Noise Ratio (SNR): Assess the quality of the generated audio by comparing the signal power to the noise power.

- Listening Test: Conduct a subjective evaluation by having listeners rate the quality and similarity of the generated speech to the original voice.

## License

This project is licensed under the [MIT License](LICENSE).

---

You can create a new file named `README.md` and copy the above content into it. Make sure to replace `[source_link]` with the actual link to download the VoxCeleb-1 dataset if applicable. Additionally, you may modify the content to suit your specific model and requirements.

Remember to include any other necessary information, such as model hyperparameters, further instructions, or any additional files required to run the model in the README.
