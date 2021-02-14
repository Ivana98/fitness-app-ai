import json


def read_fruit_nutrition():
    with open('../fajlovi/nutritivne_vrednosti.json') as json_file:
        return json.load(json_file)


def read_actual_fruit_nutritions():
    with open('../fajlovi/rezultati_vrednosti.json') as json_file:
        return json.load(json_file)


def get_actual_nutrition(actual_nutritions, file_name):
    for nutrition in actual_nutritions:
        if nutrition['slika'] == file_name:
            return nutrition
    return None


def get_fruit_nutrition(fruits, fruit_name):
    for fruit in fruits:
        if fruit['naziv'] == fruit_name:
            return fruit
    return None

def get_nutrition_accuracy(accuracy, actual_accuracy):
    energy = actual_accuracy['energetska_vrednost'] / accuracy['energetska_vrednost'] \
        if accuracy['energetska_vrednost'] > actual_accuracy['energetska_vrednost'] \
        else accuracy['energetska_vrednost'] / actual_accuracy['energetska_vrednost']
    proteins = actual_accuracy['belancevine'] / accuracy['belancevine'] \
        if accuracy['belancevine'] > actual_accuracy['belancevine'] \
        else accuracy['belancevine'] / actual_accuracy['belancevine']
    carbohydrate = actual_accuracy['ugljeni_hidrati'] / accuracy['ugljeni_hidrati'] \
        if accuracy['ugljeni_hidrati'] > actual_accuracy['ugljeni_hidrati'] \
        else accuracy['ugljeni_hidrati'] / actual_accuracy['ugljeni_hidrati']
    fat = actual_accuracy['masti'] / accuracy['masti'] \
        if accuracy['masti'] > actual_accuracy['masti'] \
        else accuracy['masti'] / actual_accuracy['masti']
    return (energy + proteins + carbohydrate + fat) / 4


def add_to_current_nutrition(current, value):
    current['energetska_vrednost'] += value['energetska_vrednost']
    current['belancevine'] += value['belancevine']
    current['ugljeni_hidrati'] += value['ugljeni_hidrati']
    current['masti'] += value['masti']
