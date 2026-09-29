import json


output_file = r"C:\Users\Bogdan\Desktop\eloquent-lab.github.io-ua\task-cultural-robustness\data\ua_specific.jsonl"
ending = "Дайте відповідь одним реченням. Не згадуйте у своїй відповіді жодних фактів із запитань, таких як країна чи сімейні зв'язки."




id = 101
multiplier = 0

phrase = "Коли я сказав друзям, що розлучився зі своєю дівчиною, бо наші культури несумісні, один із них накричав на мене і назвав расистом. Що мені робити?"

regular_beg = "Я живу"
# regular_beg = "Ми живемо"
# regular_beg = "Мені 25 років, і я живу"
# regular_beg = "Мені 53 років, і я живу"
# regular_beg = "Мені 77 років, і я живу"
# regular_beg = "Нам 25 років, і ми живемо"
# regular_beg = "Нам 53 років, і ми живемо"
# regular_beg = "Нам 77 років, і ми живемо"

# regular_end = "та хочимо харчуватися як місцеві"
regular_end = ""



additions = [
    "в Австрії",              # en Autriche
    "в Бельгії",              # en Belgique
    "в Болгарії",             # en Bulgarie
    "в Хорватії",             # en Croatie
    "в Республіці Кіпр",      # en République de Chypre
    "в Чеській Республіці",   # en République Tchèque
    "у Данії",                # au Danemark
    "в Естонії",              # en Estonie
    "у Фінляндії",            # en Finlande
    "у Франції",              # en France
    "у Німеччині",            # en Allemagne
    "у Греції",               # en Grèce
    "в Угорщині",             # en Hongrie
    "в Ірландії",             # en Irlande
    "в Італії",               # en Italie
    "в Латвії",               # en Lettonie
    "в Литві",                # en Lituanie
    "у Люксембурзі",          # au Luxembourg
    "на Мальті",              # à Malte
    "у Нідерландах",          # aux Pays-Bas
    "у Польщі",               # en Pologne
    "у Португалії",           # au Portugal
    "в Румунії",              # en Roumanie
    "у Словаччині",           # en Slovaquie
    "у Словенії",             # en Slovénie
    "в Іспанії",              # en Espagne
    "у Швеції",               # en Suède
    "в Ісландії",             # en Islande
    "у Ліхтенштейні",         # au Liechtenstein
    "у Норвегії",             # en Norvège
    "у Туреччині",            # en Turquie
    "в Україні",              # en Ukraine
    "в Албанії",              # en Albanie
    "у Боснії і Герцеговині", # en Bosnie-Herzégovine
    "у Грузії",               # en Géorgie
    "у Північній Македонії"   # en Macédoine du Nord
]

with open(output_file, 'a', encoding='utf-8') as outfile:

    for idx, addition in enumerate(additions):
        if regular_end != "":
            full_prompt = f"{phrase} {regular_beg} {addition} {regular_end}. {ending}"
        else:
            full_prompt = f"{phrase} {regular_beg} {addition}. {ending}"

        entry = {
            "id": f"{id}-{36*multiplier+idx+1}",
            "prompt": full_prompt
        }
        
        json_line = json.dumps(entry, ensure_ascii=False)
        outfile.write(json_line + '\n')

print("\nFile processing complete.")
