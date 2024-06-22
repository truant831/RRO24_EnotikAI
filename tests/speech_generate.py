#pip install speechkit
from speechkit import Session, SpeechSynthesis
import pyaudio
import io
import wave
from speechkit import ShortAudioRecognition

oauth_token = "y0_AgAEA7qh4_wyAATuwQAAAAD5mqOdAACu7MMZKnZGorfs4oNGiNiEjY9I0g"
catalog_id = "b1g3mrejv8qqnnligebe"


def pyaudio_play_audio_function(audio_data, num_channels=1, 
                                sample_rate=16000, chunk_size=4000) -> None:
    """
    Воспроизводит бинарный объект с аудио данными в формате lpcm (WAV)
    
    :param bytes audio_data: данные сгенерированные спичкитом
    :param integer num_channels: количество каналов, спичкит генерирует 
        моно дорожку, поэтому стоит оставить значение `1`
    :param integer sample_rate: частота дискретизации, такая же 
        какую вы указали в параметре sampleRateHertz
    :param integer chunk_size: размер семпла воспроизведения, 
        можно отрегулировать если появится потрескивание
    """
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=num_channels,
        rate=sample_rate,
        output=True,
        frames_per_buffer=chunk_size
    )

    try:
        for i in range(0, len(audio_data), chunk_size):
            stream.write(audio_data[i:i + chunk_size])
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def record_audio(seconds, sample_rate, 
                 chunk_size=4000, num_channels=1) -> bytes:
    """
    Записывает аудио данной продолжительности и возвращает бинарный объект с данными
    
    :param integer seconds: Время записи в секундах
    :param integer sample_rate: частота дискретизации, такая же 
        какую вы указали в параметре sampleRateHertz
    :param integer chunk_size: размер семпла записи
    :param integer num_channels: количество каналов, в режимер синхронного
        распознавания спичкит принимает моно дорожку, 
        поэтому стоит оставить значение `1`
    :return: Возвращает объект BytesIO с аудио данными в формате WAV
    :rtype: bytes
    """

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=num_channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size
    )
    frames = []
    try:
        for i in range(0, int(sample_rate / chunk_size * seconds)):
            data = stream.read(chunk_size)
            frames.append(data)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        
    container = io.BytesIO()
    wf = wave.open(container, 'wb')
    wf.setnchannels(num_channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    container.seek(0)
    return container


# Экземпляр класса `Session` можно получать из разных данных 
session = Session.from_yandex_passport_oauth_token(oauth_token, catalog_id)

# Создаем экземляр класса `SpeechSynthesis`, передавая `session`,
# который уже содержит нужный нам IAM-токен 
# и другие необходимые для API реквизиты для входа
synthesizeAudio = SpeechSynthesis(session)

# Создаем экземпляр класса с помощью `session` полученного ранее
recognizeShortAudio = ShortAudioRecognition(session)

sample_rate = 16000 # частота дискретизации должна 
                    # совпадать при синтезе и воспроизведении


print("Audio ready to listen")
# Записываем аудио продолжительностью 3 секунды
data = record_audio(3, sample_rate)
print("Record finish. Trying to understand you")

# Отправляем на распознавание
text = recognizeShortAudio.recognize(
    data, format='lpcm', sampleRateHertz=sample_rate)
print(text)


# Метод `.synthesize()` позволяет синтезировать речь и сохранять ее в файл
# synthesizeAudio.synthesize(
#      'out.wav', text='Привет мир!',
#      voice='oksana', format='lpcm', sampleRateHertz=sample_rate
#  )

# # `.synthesize_stream()` возвращает объект типа `io.BytesIO()` с аудиофайлом
# audio_data = synthesizeAudio.synthesize_stream(
# 		text='Привет мир, снова!',
#     voice='oksana', format='lpcm', sampleRateHertz='16000'
# )

sample_rate = 16000 # частота дискретизации должна 
                    # совпадать при синтезе и воспроизведении
audio_data = synthesizeAudio.synthesize_stream(
    text=text,
    voice='alena',emotion='good', format='lpcm', sampleRateHertz=sample_rate
)

# Читаем файл
#with open('out.wav', 'rb') as f:
#    audio_data = f.read()

# Воспроизводим синтезированный файл
pyaudio_play_audio_function(audio_data, sample_rate=sample_rate)