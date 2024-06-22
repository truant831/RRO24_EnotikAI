
from speechkit import Session, SpeechSynthesis
import pyaudio

oauth_token = "y0_AgAEA7qh4_wyAATuwQAAAAD5mqOdAACu7MMZKnZGorfs4oNGiNiEjY9I0g"
catalog_id = "b1g3mrejv8qqnnligebe"


# Экземпляр класса `Session` можно получать из разных данных 
session = Session.from_yandex_passport_oauth_token(oauth_token, catalog_id)

sample_rate = 16000 # частота дискретизации должна 
                    # совпадать при синтезе и воспроизведении

# Создаем экземляр класса `SpeechSynthesis`, передавая `session`,
# который уже содержит нужный нам IAM-токен 
# и другие необходимые для API реквизиты для входа
synthesizeAudio = SpeechSynthesis(session)

names_eng={0: 'DOBBLINATOR', 1: 'Spot-It', 2: 'VS-SH', 
           3: 'axe', 4: 'backpack', 5: 'bear', 6: 'binocular', 7: 'boat', 8: 'boot',
            9: 'camp', 10: 'chair', 11: 'compass', 12: 'cone', 13: 'cup', 14: 'eagle', 
            15: 'fire', 16: 'first_aid_kit', 17: 'fish', 18: 'flashlight', 19: 'frog', 
            20: 'giutar', 21: 'glasses', 22: 'hammcok', 23: 'harmonica', 24: 'hotdog', 
            25: 'house', 26: 'house on wheel', 27: 'ice_box', 28: 'incekt', 29: 'kettle', 
            30: 'knife', 31: 'kumbaya', 32: 'lamp', 33: 'leafes', 
            34: 'lyceum-innopolis', 35: 'man', 36: 'map', 
            37: 'matchess', 38: 'mini_flashlight', 39: 'moon', 40: 'moose', 41: 'mushrooms', 
            42: 'music', 43: 'nuts', 44: 'owl', 45: 'paw', 46: 'raccoon', 47: 'radio', 
            48: 'sandwitch', 49: 'shashlik', 50: 'sign', 51: 'sleeping_bag', 52: 'spray', 53: 'stick', 
            54: 'sun', 55: 'table', 56: 'tent', 57: 'thermos', 58: 'tree', 59: 'undefined', 60: 'waterfall', 
            61: 'wood'}

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
       54: 'солнце', 55: 'стол', 56: 'палатка', 57: 'термос', 58: 'дерево',59: 'неведома зверушка', 60: 'водопад', 
       61: 'палено'}

phrazes={"listen":"Слушаю Вас!","robot_win":"ХА-Ха-ха! я победил тебя, кожанный мешок!","nobody":"В этот раз тебе повезло! Ничья!","human_win":"как тебе это удалось, белковое создание? ты победил!"}

for i in range(len(names)):
    #Метод `.synthesize()` позволяет синтезировать речь и сохранять ее в файл
    print(str(i)+'.wav', names[i])
    synthesizeAudio.synthesize(
        str(i)+'.wav', text='ДОББЛЬ!! '+names[i]+"!",
        voice='alena', emotion='good', format='lpcm', sampleRateHertz=sample_rate
    )

# for key in phrazes:
#     #Метод `.synthesize()` позволяет синтезировать речь и сохранять ее в файл
#     print(key+'.wav', phrazes[key])
#     synthesizeAudio.synthesize(
#         key+'.wav', text=phrazes[key],
#         voice='alena', emotion='good', format='lpcm', sampleRateHertz=sample_rate
#    )