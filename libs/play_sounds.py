import pyaudio
import time

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
    #Прочитать и может переделать на callback https://github.com/raspberrypi/linux/issues/994, https://people.csail.mit.edu/hubert/pyaudio/
    # diffrent libs to play sounds: https://realpython.com/playing-and-recording-sound-python/#pyaudio
    #можно попробовать использовать https://pypi.org/project/sounddevice/
    #или alsaaudio/pyalsaaudio https://github.com/larsimmisch/pyalsaaudio/blob/master/playwav.py
    try:
        for i in range(0, len(audio_data), chunk_size):
            stream.write(audio_data[i:i + chunk_size])
    finally:
        # Wait the stream end                                                                                                              
        time.sleep(1.0)
        #stop and kill stream
        stream.stop_stream()
        stream.close()
        p.terminate()

sample_rate = 16000 # частота дискретизации должна 
                    # совпадать при синтезе и воспроизведении


names={0: 'DOBBLINATOR', 1: 'Spot-It', 2: 'ВСОШ',
       3: 'топор', 4: 'рюкзак', 5: 'медведь', 6: 'бинокль', 7: 'лодка', 8: 'ботинок', 
       9: 'camp', 10: 'стул', 11: 'компасс', 12: 'шишка', 13: 'чашка', 14: 'орёл', 
       15: 'костер', 16: 'аптечка', 17: 'рыба', 18: 'фонарик', 19: 'лягушка', 
       20: 'гитара', 21: 'очки', 22: 'гамак', 23: 'гармошка', 24: 'hotdog', 
       25: 'дом', 26: 'дом на колесах', 27: 'коробка со льдом', 28: 'комар', 29: 'чайник', 
       30: 'нож', 31: 'kumbaya', 32: 'лампа', 33: 'листья', 
       34: "лицей иннополис", 35: 'человек', 36: 'карта', 
       37: 'спички', 38: 'налобный фонарь', 39: 'луна', 40: 'олень', 41: 'грибы', 
       42: 'магнитофон', 43: 'орешки', 44: 'сова', 45: 'след', 46: 'Енотик (мой любимый зверек)', 47: 'радио', 
       48: 'сэндвич', 49: 'жареный зефир', 50: 'знак', 51: 'спальник', 52: 'спрэй', 53: 'палка', 
       54: 'солнце', 55: 'стол', 56: 'палатка', 57: 'термос', 58: 'дерево', 60: 'водопад', 
       61: 'палено'}


def Say_card_name(index):
    # Читаем файл
    print(str(index)+'.wav', names[index])
    with open("sounds/"+str(index)+'.wav', 'rb') as f:
       audio_data = f.read()

    pyaudio_play_audio_function(audio_data, sample_rate=sample_rate)

def Say_phraze(template):
    # Читаем файл
    with open("sounds/"+str(template)+'.wav', 'rb') as f:
       audio_data = f.read()
    pyaudio_play_audio_function(audio_data, sample_rate=sample_rate)


#Say_card_name(24)
#phraza="Слушаю Вас"
# if score_robot>score_robot:
#     Say_phraze("robot_win")
# else if score_robot==score_human:
#     Say_phraze("nobody")
# else:
#     Say_phraze("human_win")