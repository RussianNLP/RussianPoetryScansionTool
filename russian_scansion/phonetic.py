"""
Фонетический словарь (ударятор).
При выполнении словарь будет загружен из текстовых файлов, подготовлен и сохранен в pickle-файле
для последующей быстрой подгрузки.

19.05.2022 Добавлено автоисправление некоторых орфографических ошибок типа "тишына", "стесняцца"
07.06.2022 Добавлено автоисправление децкий ==> детский
02.08.2022 Исправление опечатки - твердый знак вместо мягкого "пъянки"
11.08.2022 Добавлена продуктивная приставка "супер-"
12.08.2022 Добавлена продуктивная приставка "лже-"
24.01.2023 Добавлен фильтр для отсева бракованных лексем в качестве ключей в слове ударений
"""

import json
import traceback
import typing
import functools

import yaml
import pickle
import os
import io
import logging
import re
import huggingface_hub

from accentuator import AccentuatorWrapper

import rusyllab



def is_good_stress_key(word):
    if re.search(r'[a-z0-9·]', word, flags=re.I) is not None:
        #print('DEBUG@37: ', word)
        return False
    if len(word) < 2:
        #print('DEBUG@40: ', word)
        return False

    m1 = re.search(r'([^абвгдеёжзийклмнопрстуфхцчшщъыьэюя\-̀́])', word, flags=re.I)
    if m1 is not None:
        # В слове не должно быть символов, кроме кириллицы, дефиса и значков ударения.
        #print('DEBUG@45: word={} char={}'.format(word, m1.group(1)))
        return False

    if any((c.lower() not in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя̀́') for c in [word[0], word[-1]]):
        # первый и последний символ слова обязаны быть кириллицей или значком ударения, а не пунктуаторами или типа того
        #print('DEBUG@50: ', word)
        return False
    return True



class WordAccentuation(object):
    def __init__(self, stress_pos: int, secondary_accentuation: typing.List[int]):
        self.stress_pos = stress_pos
        self.secondary_accentuation = secondary_accentuation

    @staticmethod
    def build_nostress():
        return WordAccentuation(-1, None)

    @staticmethod
    def build_stress1():
        """Для слов с единственной гласной, которая будет ударной"""
        return WordAccentuation(1, None)


class Accents:
    def __init__(self, device="cuda"):
        self.device = device
        self.ambiguous_accents = None
        self.ambiguous_accents2 = None
        self.word_accents_dict = None
        self.yo_words = None
        self.yo_dict = None
        self.rhymed_words = set()
        self.allow_rifmovnik = False

        self.allow_nonrussian_accentuation = False
        self.re_cyrlocator = re.compile(r'[абвгдеёжзийклмнопрстуфхцчшщъыьэюя]', flags=re.I)
        self.fuzzy_rhyming_cache = dict()

    def sanitize_word(self, word):
        return word.lower() #.replace(u'ё', u'е')

    def load(self, data_dir, all_words):
        # Рифмовник для нечеткой рифмы
        with open(os.path.join(data_dir, 'rifmovnik.small.upgraded.json'), 'r') as f:
            rhyming_data = json.load(f)
            self.rhyming_dict = dict((key, values) for key, values in rhyming_data['dictionary'].items() if len(values) > 0)

        # пары слов, который будем считать рифмующимися
        with io.open(os.path.join(data_dir, 'rhymed_words.txt'), 'r', encoding='utf-8') as rdr:
            for line in rdr:
                s = line.strip()
                if s and not s.startswith('#'):
                    i = s.index(' ')
                    word1 = s[:i].strip()
                    word2 = s[i+1:].strip()
                    self.rhymed_words.add((word1, word2))

        with open(os.path.join(data_dir, 'prefix_derivation.json')) as f:
            self.derivation_data = json.load(f)

        self.yo_words = dict()

        # импорт однозначной ёфикации
        # with open(os.path.join(data_dir, 'yo_3.json'), 'r') as f:
        #     data = json.load(f)
        #     for word, yo in data.items():
        #         self.yo_words[word] = yo.lower()

        # однозначная ёфикация
        # for fn in ['solarix_yo.txt', 'yo_2.txt']:
        #     path = os.path.join(data_dir, fn)
        #     logging.info('Loading words with ё from "%s"', path)
        #     with io.open(path, 'r', encoding='utf-8') as rdr:
        #         for line in rdr:
        #             word = line.strip().lower()
        #             key = word.replace('ё', 'е')
        #             self.yo_words[key] = word

        for fn in ['yo_words.txt']:
            path = os.path.join(data_dir, fn)
            logging.info('Loading words with ё from "%s"', path)
            with io.open(path, 'r', encoding='utf-8') as rdr:
                for line in rdr:
                    word = line.strip().lower()
                    if 'ё' in word:
                        key = word.replace('ё', 'е')
                        self.yo_words[key] = word

        # словарь для разрешения случаев ёфикации, зависящей от грамматической формы слова.
        path = os.path.join(data_dir, 'yo_by_gram.json')
        with open(path, 'r') as f:
            self.yo_dict = json.load(f)

        # Информация о словах, которые для разных грамматических форм могут давать разное ударение.
        path = os.path.join(data_dir, 'ambiguous_accents.yaml')
        logging.info('Loading ambiguous accents information from "%s"', path)
        d = yaml.safe_load(io.open(path, 'r', encoding='utf-8').read())

        # 03.08.2022
        # В словаре встречаются вхождения "Case=Every"
        # Раскроем, чтобы было явное перечисление всех падежей.
        d2 = dict()
        for entry_name, entry_data in d.items():
            if any((c in entry_name[0]) for c in '-.~'):
                continue

            if 'ӂ' in entry_name:
                entry_name = entry_name.replace('ӂ', 'ж')

            if entry_name.startswith('нэйро'):
                continue

            entry_data2 = dict()
            for form, tagsets in entry_data.items():
                tagsets2 = []
                for tagset in tagsets:
                    if 'Case=Every' in tagset:
                        for case in ['Nom', 'Gen', 'Ins', 'Acc', 'Dat', 'Loc']:
                            tagset2 = tagset.replace('Case=Every', 'Case={}'.format(case))
                            tagsets2.append(tagset2)
                    else:
                        tagsets2.append(tagset)

                entry_data2[form] = tagsets2

            d2[entry_name] = entry_data2

        self.ambiguous_accents = d2

        # 14.02.2022 сделаем проверку содержимого, чтобы не словить ошибку в рантайме.
        for word, wdata in self.ambiguous_accents.items():
            for stressed_form, tagsets in wdata.items():
                if not any((c in 'АЕЁИОУЫЭЮЯ') for c in stressed_form):
                    print('Missing stressed vowel in "ambiguous_accents.yaml" for word={}'.format(word))
                    exit(0)

        logging.info('%d items in ambiguous_accents', len(self.ambiguous_accents))

        # Некоторые слова допускают разное ударение для одной грамматической формы: пОнял-понЯл
        # 14.06.2023 сразу конвертируем человекочитабельные записи в позиции ударения.
        self.ambiguous_accents2 = dict()
        for fn in ['synthetic_ambiguous_stresses.yaml', 'ambiguous_accents_2.yaml']:
            for word, accents in yaml.safe_load(io.open(os.path.join(data_dir, fn), 'r', encoding='utf-8').read()).items():
                stress_posx = []
                for accent in accents:
                    stress_pos = -1
                    n_vowels = 0
                    for c in accent:
                        if c.lower() in 'уеыаоэёяию':
                            n_vowels += 1

                        if c in 'АЕЁИОУЫЭЮЯ':
                            stress_pos = n_vowels
                            break

                    if stress_pos == -1:
                        raise ValueError('Could not find stressed position in word "{}" in ambiguous_accents2'.format(accent))

                    stress_posx.append(stress_pos)
                self.ambiguous_accents2[word] = stress_posx

        self.secondary_stress_dict = dict()
        for fn in ['вторичные ударения на приставках 2.txt', 'добавка вторичных ударений для прилагательных с приставкой НЕ.txt', 'новые составные слова с вторичным ударением.txt', 'wiktionary_secondary_stress.txt', 'secondary_stress.txt']:
            logging.info('Loading secondary stress information from "%s"', fn)
            with io.open(os.path.join(data_dir, fn), 'r', encoding='utf-8') as rdr:
                for line in rdr:
                    word = line.strip()
                    sx = []
                    for c in word:
                        if c in 'АЕЁИОУЫЭЮЯ':
                            sx.append(2)  # вторичное ударение на этом слоге
                        elif c in 'аеёиоуыэюя':
                            sx.append(0)  # нет ударения или основное ударение
                    self.secondary_stress_dict[word.lower()] = tuple(sx)

        self.word_accents_dict = dict()
        # if True:
        #     path = os.path.join(data_dir, 'single_accent.dat')
        #     logging.info('Loading stress information from "%s"', path)
        #     with io.open(path, 'r', encoding='utf-8') as rdr:
        #         for line in rdr:
        #             tx = line.split('\t')
        #             if len(tx) == 2:
        #                 word, accent = tx[0], tx[1]
        #                 if is_good_stress_key(word):
        #                     n_vowels = 0
        #                     for c in accent:
        #                         if c.lower() in 'уеыаоэёяию':
        #                             n_vowels += 1
        #                             if c.isupper():
        #                                 stress = n_vowels
        #                                 self.word_accents_dict[word.lower()] = stress
        #                                 break

        # if True:
        #     path2 = os.path.join(data_dir, 'accents.txt')
        #     logging.info('Loading stress information from "%s"', path2)
        #     with codecs.open(path2, 'r', 'utf-8') as rdr:
        #         for line in rdr:
        #             tx = line.strip().split('#')
        #             if len(tx) == 2:
        #                 forms = tx[1].split(',')
        #                 for form in forms:
        #                     word = self.sanitize_word(form.replace('\'', '').replace('`', ''))
        #                     if is_good_stress_key(word):
        #                         if all_words is None or word in all_words:
        #                             if '\'' in form:
        #                                 accent_pos = form.index('\'')
        #                                 nb_vowels_before = self.get_vowel_count(form[:accent_pos], abbrevs=False)
        #                                 if word not in self.word_accents_dict:
        #                                     self.word_accents_dict[word] = nb_vowels_before
        #                             elif 'ё' in form:
        #                                 accent_pos = form.index('ё')
        #                                 nb_vowels_before = self.get_vowel_count(form[:accent_pos], abbrevs=False)+1
        #                                 if word not in self.word_accents_dict:
        #                                     self.word_accents_dict[word] = nb_vowels_before

        # if True:
        #     stress_char = '́'
        #     stress2_char = '̀'
        #     p3 = os.path.join(data_dir, 'ruwiktionary-accents.txt')
        #     logging.info('Loading stress information from "%s"', p3)
        #     with codecs.open(p3, 'r', 'utf-8') as rdr:
        #         for iline, line in enumerate(rdr):
        #             word = line.strip()
        #
        #             # 17.05.2023 заменяем символ "'" на юникодный модификатор ударения
        #             if "'" in word and '\u0301' not in word:
        #                 word = word.replace("'", '\u0301')
        #
        #             if is_good_stress_key(word):
        #                 nword = word.replace(stress_char, '').replace('\'', '').replace('ѝ', 'и').replace('ѐ', 'е').replace(stress2_char, '').lower()
        #                 if len(nword) > 2:
        #                     if stress_char in word:
        #                         accent_pos = word.index(stress_char)
        #                         nb_vowels_before = self.get_vowel_count(word[:accent_pos], abbrevs=False)
        #                         if nword not in self.word_accents_dict:
        #                             self.word_accents_dict[nword] = nb_vowels_before
        #                     elif '\'' in word:
        #                         accent_pos = word.index('\'')
        #                         nb_vowels_before = self.get_vowel_count(word[:accent_pos], abbrevs=False)
        #                         if nword not in self.word_accents_dict:
        #                             self.word_accents_dict[nword] = nb_vowels_before
        #                     elif 'ё' in word:
        #                         accent_pos = word.index('ё')
        #                         nb_vowels_before = self.get_vowel_count(word[:accent_pos], abbrevs=False)
        #                         stress_pos = nb_vowels_before + 1
        #                         if nword not in self.word_accents_dict:
        #                             self.word_accents_dict[nword] = stress_pos

        # if True:
        #     path = os.path.join(data_dir, 'words_accent.json')
        #     logging.info('Loading stress information from "%s"', path)
        #     d = json.loads(open(path).read())
        #     for word, a in d.items():
        #         if is_good_stress_key(word):
        #             nword = self.sanitize_word(word)
        #             if nword not in self.word_accents_dict:
        #                 self.word_accents_dict[nword] = a

        true_accent_entries = dict()
        # accentuated_words.txt это основной словарь для случаев, когда ударение в слове фиксировано.
        # true_accents.txt это словарь для удобного добавления слов в accentuated_words.txt.
        for fn in ['новые слова с ударением из википедии.txt', 'accentuated_words.txt', 'true_accents.txt']:
            logging.info('Loading stress information from "%s"', fn)
            with io.open(os.path.join(data_dir, fn), 'r', encoding='utf-8') as rdr:
                for line in rdr:
                    word = line.strip()
                    if word:
                        nword = self.sanitize_word(word)
                        if nword in self.ambiguous_accents:
                            del self.ambiguous_accents[nword]
                        m = re.search('([АЕЁИОУЭЮЯЫ])', word)
                        if m is None:
                            logging.error('Invalid item "%s" in "%s"', word, fn)
                            exit(0)

                        accent_char = m.groups(0)[0]
                        accent_pos = word.index(accent_char)
                        nb_vowels_before = self.get_vowel_count(word[:accent_pos], abbrevs=False) + 1

                        # Детектируем переопеределение ударения в слове. Такие слова с неоднозначным ударением
                        # надо переносить в ambiguous_accents_2.yaml
                        if nword in true_accent_entries and true_accent_entries[nword] != word:
                            logging.error('Controversial redefenition of stress position for word "%s" in "%s": %s and %s', nword, fn, true_accent_entries[nword], word)
                            exit(0)

                        self.word_accents_dict[nword] = nb_vowels_before
                        true_accent_entries[nword] = word

        if 'ль' in self.word_accents_dict:
            del self.word_accents_dict['ль']  # в файле "words_accent.json" зачем-то прописано ударение для частицы "ль", удалим

        logging.info('%d items in word_accents_dict', len(self.word_accents_dict))

    def save_pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.ambiguous_accents, f)
            pickle.dump(self.ambiguous_accents2, f)
            pickle.dump(self.word_accents_dict, f)
            pickle.dump(self.secondary_stress_dict, f)
            pickle.dump(self.yo_words, f)
            pickle.dump(self.yo_dict, f)
            pickle.dump(self.rhymed_words, f)
            pickle.dump(self.rhyming_dict, f)
            pickle.dump(self.derivation_data, f)

    def load_pickle(self, path):
        with open(path, 'rb') as f:
            self.ambiguous_accents = pickle.load(f)
            self.ambiguous_accents2 = pickle.load(f)
            self.word_accents_dict = pickle.load(f)
            self.secondary_stress_dict = pickle.load(f)
            self.yo_words = pickle.load(f)
            self.yo_dict = pickle.load(f)
            self.rhymed_words = pickle.load(f)
            self.rhyming_dict = pickle.load(f)
            self.derivation_data = pickle.load(f)

    def after_loading(self, model_name_or_path):
        self.stress_model = AccentuatorWrapper(model_name_or_path, self.device)
        self.predicted_accents = dict()

    def load_pretrained(self, model_name_or_path):
        if model_name_or_path == 'inkoziev/accentuator':
            dict_filepath = huggingface_hub.hf_hub_download(repo_id=model_name_or_path, filename='accents.pkl')
        else:
            dict_filepath = os.path.join(model_name_or_path, 'accents.pkl')
        self.load_pickle(dict_filepath)
        self.after_loading(model_name_or_path)

    def conson(self, c1):
        # Оглушение согласной
        if c1 == 'б':
            return 'п'
        elif c1 == 'в':
            return 'ф'
        elif c1 == 'г':
            return 'к'
        elif c1 == 'д':
            return 'т'
        elif c1 == 'ж':
            return 'ш'
        elif c1 == 'з':
            return 'с'

        return c1

    def get_secondary_accentuation(self, word: str):
        return self.secondary_stress_dict.get(word)

    def yoficate(self, word):
        return self.yo_words.get(word, word)

    def yoficate2(self, word, ud_tags):
        if word in self.ambiguous_accents2:
            return word

        if word in self.yo_words:
            return self.yo_words[word]

        if ud_tags is not None and word in self.yo_dict:
            best_word2 = None
            best_matching = 0
            ud_tagset = set(ud_tags)
            for word2, tagsets in self.yo_dict[word].items():
                for tagset in tagsets:
                    tx = set(tagset.split('|'))
                    nb_matched = len(ud_tagset.intersection(tx))
                    if nb_matched > best_matching:
                        best_matching = nb_matched
                        best_word2 = word2

            if best_word2 is not None:
                return best_word2

        return word

    def pronounce_full(self, word):
        # Фонетическая транскрипция всего слова
        # Сейчас сделана затычка в виде вызова транскриптора окончания, но
        # это неправильно с точки зрения обработки ударных/безударных гласных.
        return self.pronounce(self.yoficate(word))

    def pronounce(self, s):
        if s is None or len(s) == 0:
            return ''

        # Фонетическая транскрипция фрагмента слова, НАЧИНАЯ С УДАРНОЙ ГЛАСНОЙ
        #                                            ^^^^^^^^^^^^^^^^^^^^^^^^^
        if s.endswith('жь'):
            # РОЖЬ -> РОЖ
            s = s[:-1]
        elif s.endswith('шь'):
            # МЫШЬ -> МЫШ
            s = s[:-1]
        elif s.endswith('щь'):
            # МОРЩЬ ==> МОРЩ
            s = s[:-1]

        # СОЛНЦЕ -> СОНЦЕ
        s = s.replace('лнц', 'нц')

        # СЧАСТЬЕ -> ЩАСТЬЕ
        s = s.replace('сч', 'щ')

        # БРАТЬСЯ -> БРАЦА
        s = s.replace('ться', 'ца')

        # БОЯТСЯ -> БОЯЦА
        s = s.replace('тся', 'ца')

        # БРАТЦЫ -> БРАЦА
        s = s.replace('тц', 'ц')

        # ЖИР -> ЖЫР
        s = s.replace('жи', 'жы')

        # ШИП -> ШЫП
        s = s.replace('ши', 'шы')

        # МОЦИОН -> МОЦЫОН
        s = s.replace('ци', 'цы')

        # ЖЁСТКО -> ЖОСТКО
        s = s.replace('жё', 'жо')

        # ОКОНЦЕ -> ОКОНЦЭ
        s = s.replace('це', 'цэ')

        # БЕЗБРАЧЬЯ
        if 'чь' in s:
            s = s.replace('чья', 'ча')
            s = s.replace('чье', 'чэ')
            s = s.replace('чьё', 'чо')
            s = s.replace('чью', 'чё')

        # двойные согласные:
        # СУББОТА -> СУБОТА
        s = re.sub(r'([бвгджзклмнпрстфхцчшщ])\1', r'\1', s)

        # оглушение:
        # СКОБКУ -> СКОПКУ
        new_s = []
        for c1, c2 in zip(s, s[1:]):
            if c2 in 'кпстфх':
                new_s.append(self.conson(c1))
            else:
                new_s.append(c1)

        # последнюю согласную оглушаем всегда:
        # ГОД -> ГОТ
        new_s.append(self.conson(s[-1]))

        s = ''.join(new_s)

        # огрушаем последнюю согласную с мягким знаком:
        # ВПРЕДЬ -> ВПРЕТЬ
        if len(s) >= 2 and s[-1] == 'ь' and s[-2] in 'бвгдз':
            s = s[:-2] + self.conson(s[-2]) + 'ь'

        if self.get_vowel_count(s, abbrevs=False) > 1:
            for ic, c in enumerate(s):
                if c in "уеыаоэёяию":
                    # нашли первую, ударную гласную
                    new_s = s[:ic+1]
                    for c2 in s[ic+1:]:
                        # безударные О меняем на А (потом надо бы ввести фонетический алфавит)
                        if c2 == 'о':
                            new_s += 'а'
                        else:
                            new_s += c2

                    s = new_s
                    break

            # ослабляем безударную Е
            # МУСУЛЬМАНЕ = МУСУЛЬМАНИ
            #if s[-1] == 'е':
            #    s = s[:-1] + 'и'

        return s

    def get_vowel_count(self, word0, abbrevs=True):
        word = self.sanitize_word(word0)
        vowels = "уеыаоэёяиюaeoy" # "уеыаоэёяию"   28.12.2021 добавил гласные из латиницы
        vowel_count = 0

        for ch in word:
            if ch in vowels:
                vowel_count += 1

        if vowel_count == 0 and len(word0) > 1 and abbrevs:
            if word.lower() in ['ль']:
                return 0

            # аббревиатуры из согласных: ГКЧП
            # Считаем, что там число гласных=длине: Гэ-Ка-Че-Пэ
            return len(word0)

        return vowel_count

    def is_oov(self, word):
        return 'ё' not in word and word not in self.word_accents_dict and\
            word not in self.ambiguous_accents and\
            word not in self.ambiguous_accents2 and\
            word not in self.yo_words and\
            word not in self.yo_dict

    def get_ambiguous_stresses1(self, word):
        stress_positions = []
        if word in self.ambiguous_accents:
            for accented, tagsets in self.ambiguous_accents[word].items():
                n_vowels = 0
                for c in accented:
                    if c.lower() in 'уеыаоэёяию':
                        n_vowels += 1
                        if c.isupper():
                            stress_positions.append(n_vowels)

        return stress_positions

    def get_ambiguous_stresses2(self, word):
        if word in self.ambiguous_accents2:
            return self.ambiguous_accents2[word]
        else:
            return self.get_ambiguous_stresses1(word)

    def predict_ambiguous_accent(self, word, ud_tags):
        best_accented = None
        best_matching = 0
        ud_tagset = set(ud_tags)
        for accented, tagsets in self.ambiguous_accents[word].items():
            for tagset in tagsets:
                tx = set(tagset.split('|'))
                nb_matched = len(ud_tagset.intersection(tx))
                if nb_matched > best_matching:
                    best_matching = nb_matched
                    best_accented = accented

        if best_accented is None:
            #print('ERROR@166 Could not choose accent for word="{}" ud_tags="{}"'.format(word, '|'.join(ud_tags)))
            #exit(0)
            return -1

        n_vowels = 0
        for c in best_accented:
            if c.lower() in 'уеыаоэёяию':
                n_vowels += 1
                if c.isupper():
                    return n_vowels

        msg = 'Could not predict stress position in word="{}" tags="{}"'.format(word, ' '.join(ud_tags) if ud_tags else '[]')
        raise ValueError(msg)

    def get_all_stress_variants(self, word):
        variants = []

        if word in self.ambiguous_accents2:
            for stress_pos in self.ambiguous_accents2[word]:
                variants.append((stress_pos, 'ambiguous_accents2'))
        elif word in self.ambiguous_accents:
            for accent in self.ambiguous_accents[word].keys():
                stress_pos = -1
                n_vowels = 0
                for c in accent:
                    if c.lower() in 'уеыаоэёяию':
                        n_vowels += 1

                    if c in 'АЕЁИОУЫЭЮЯ':
                        stress_pos = n_vowels
                        break

                variants.append((stress_pos, 'ambiguous_accents'))
        elif word in self.word_accents_dict:
            stress_pos = self.word_accents_dict[word]
            variants.append((stress_pos, 'word_accents_dict'))
        else:
            char_index = self.predict_stressed_charpos(word)
            n_vowels = 0
            stress_pos = 0
            for i, c in enumerate(word):
                if c.lower() in 'уеыаоэёяию':
                    n_vowels += 1

                if char_index == i:
                    stress_pos = n_vowels
                    break

            variants.append((stress_pos, 'predict_stressed_charpos'))

        return variants

    def predict_stressed_charpos(self, word):
        """ Вернет индекс ударной буквы (это будет гласная, конечно). Отсчет от 0 """
        if word in self.word_accents_dict:
            vi = self.word_accents_dict[word]
            nv = 0
            for ic, c in enumerate(word):
                if c in "уеыаоэёяию":
                    nv += 1

                    if nv == vi:
                        return ic

        if re.match(r'^[бвгджзклмнпрстфхцчшщ]{2,}$', word):
            # Считаем, что в аббревиатурах, состоящих из одних согласных,
            # ударение падает на последний "слог":
            # ГКЧП -> Гэ-Ка-Че-П^э
            return len(word)

        i = self.stress_model.predict(word)
        return i

    def predict_stress(self, word):
        if word in self.predicted_accents:
            return self.predicted_accents[word]

        if re.match(r'^[бвгджзклмнпрстфхцчшщ]{2,}$', word):
            # Считаем, что в аббревиатурах, состоящих из одних согласных,
            # ударение падает на последний "слог":
            # ГКЧП -> Гэ-Ка-Че-П^э
            return len(word)

        #print('DEBUG@146 word={}'.format(word))
        i = self.stress_model.predict(word)
        #if len(i) != 1:
        #    return -1
        #i = i[0]

        # получили индекс символа с ударением.
        # нам надо посчитать гласные слева (включая ударную).
        nprev = self.get_vowel_count(word[:i], abbrevs=False)
        accent = nprev + 1
        self.predicted_accents[word] = accent
        return accent

    def get_accent0(self, word0, ud_tags=None):
        word = self.yoficate2(self.sanitize_word(word0), ud_tags)
        if 'ё' in word:
            # считаем, что "ё" всегда ударная (исключение - слово "ёфикация" и однокоренные)
            n_vowels = 0
            for c in word:
                if c in 'уеыаоэёяию':
                    n_vowels += 1
                    if c == 'ё':
                        return n_vowels

        if ud_tags and self.ambiguous_accents and word in self.ambiguous_accents:
            return self.predict_ambiguous_accent(word, ud_tags)

        if word in self.word_accents_dict:
            return self.word_accents_dict[word]

        return self.predict_stress(word)

    def get_accent(self, word0, ud_tags=None):
        word = self.sanitize_word(word0)
        word = self.yoficate2(word, ud_tags)

        vowel_count = self.get_vowel_count(word)
        if vowel_count == 0:
            if len(word) == 1:
                # единственная согласная
                return -1
            elif word.lower() == 'ль':
                return -1

        if vowel_count == 1:
            # Для слов, содержащих единственную гласную, сразу возвращаем позицию ударения на этой гласной
            return 1

        if 'ё' in word:
            if word in self.word_accents_dict:
                return self.word_accents_dict[word]

            # считаем, что "ё" всегда ударная (исключение - слово "ёфикация" и однокоренные)
            n_vowels = 0
            for c in word:
                if c in 'уеыаоэёяию':
                    n_vowels += 1
                    if c == 'ё':
                        return n_vowels

        if ud_tags and self.ambiguous_accents and word in self.ambiguous_accents:
            return self.predict_ambiguous_accent(word, ud_tags)

        if word in self.ambiguous_accents2:
            # Вообще, тут оказываться мы не должны. Надо переделывать логику работы с вариантами ударности.
            return self.ambiguous_accents2[word][0]

        if word in self.word_accents_dict:
            return self.word_accents_dict[word]

        # 19.05.2022 в порошках и т.п. по законам жанра допускаются намеренные ошибки типа "ошыбка".
        # Попробуем скорректировать такие ошибки.
        corrections = [('тьса', 'тся'),  # рвЕтьса
                       ('тьса', 'ться'),  # скрытьса
                       ('ться', 'тся'),
                       ('юцца', 'ются'),
                       ('цца', 'ться'),
                       ('юца', 'ются'),  # бьюца
                       ('шы', 'ши'), ('жы', 'жи'), ('цы', 'ци'), ('щю', 'щу'), ('чю', 'чу'),
                       ('ща', 'сча'),
                       ('щя', 'ща'),  # щями
                       ("чя", "ча"),  # чящя
                       ("жэ", "же"),  # художэственный
                       ('цэ', 'це'), ('жо', 'жё'), ('шо', 'шё'), ('чо', 'чё'), ('чьк', 'чк'),
                       ('що', 'щё'),  # вощоный
                       ('щьк', 'щк'),
                       ('цк', 'тск'),  # 07.06.2022 децкий ==> детский
                       ('цца', 'тся'),  # 04.08.2022 "льюцца"
                       ('ъе', 'ьё'),  # бъется
                       ('ье', 'ъе'),  # сьЕли
                       ('сн', 'стн'),  # грусный
                       ('цц', 'тц'),  # браццы
                       ('цц', 'дц'),  # триццать
                       ('чт', 'чьт'),  # прячте
                       ('тьн', 'тн'),  # плОтьник
                       ('зд', 'сд'),  # здачу
                       ('тса', 'тся'),  # гнУтса
                       ]

        for m2 in corrections:
            if m2[0] in word:
                word2 = word.replace(m2[0], m2[1])
                if word2 in self.word_accents_dict:
                    return self.word_accents_dict[word2]

        # восстанавливаем мягкий знак в "стоиш" "сможеш"  "сбереч"
        # встретимса
        #        ^^^
        e_corrections = [('иш', 'ишь'),  # стоиш
                         ('еш', 'ешь'),  # сможеш
                         ('еч', 'ечь'),  # сбереч
                         ('мса', 'мся'),  # встретимса
                         ]
        for e1, e2 in e_corrections:
            if word.endswith(e1):
                word2 = word[:-len(e1)] + e2
                if word2 in self.word_accents_dict:
                    return self.word_accents_dict[word2]



        # убираем финальный "ь" после шипящих:
        # клавишь
        if re.search(r'[чшщ]ь$', word):
            word2 = word[:-1]
            if word2 in self.word_accents_dict:
                return self.word_accents_dict[word2]

        # повтор согласных и гласных сокращаем до одной согласной:
        # щщупать
        if len(word) > 1:
            cn = re.search(r'(\w)\1', word, flags=re.I)
            if cn:
                c1 = cn.group(1)[0]
                try:
                    word2 = re.sub(c1+'{2,}', c1, word, flags=re.I)
                    if word2 in self.word_accents_dict:
                        return self.word_accents_dict[word2]
                except re.error:
                    print(traceback.format_exc())
                    exit(1)

        # Некоторые грамматические формы в русском языке имеют
        # фиксированное ударение.
        pos1 = word.find('ейш') # сильнейший, наимудрейшие
        if pos1 != -1:
            stress_pos = self.get_vowel_count(word[:pos1], abbrevs=False) + 1
            return stress_pos

        # TODO: !!! префиксальную деривация перевести на использование TRIE для подбора префикса !!!
        if ud_tags and 'VERB' in ud_tags:
            # Глагольная префиксальная деривация.
            for prefix in self.derivation_data['verb']['prefixes']:
                if word.startswith(prefix) and word[len(prefix):] in self.derivation_data['verb']['verb2stress']:
                    stressed_forms = self.derivation_data['verb']['verb2stress'][word[len(prefix):]]
                    stressed_form1 = list(stressed_forms)[0]  # это огрубление модели, так как могут быть случаи с 2 вариантами ударения.
                    stressed_word = prefix + stressed_form1
                    n_vowels = 0
                    for c in stressed_word:
                        if c in 'уеыаоэёяию':
                            n_vowels += 1
                        elif c == '\u0301':
                            return n_vowels
        elif ud_tags and any((pos in ud_tags) for pos in ('NOUN', 'PROPN', 'ADJ')):
            # Префиксальная деривация для существительных и прилагательных
            pos_prefixes = None
            pos2stress = None
            if 'ADJ' in ud_tags:
                pos_prefixes = self.derivation_data['adj']['prefixes'] + self.derivation_data['compound_prefixes']
                pos2stress = self.derivation_data['adj']['adj2stress']
            elif any((pos in ud_tags) for pos in ('NOUN', 'PROPN')):
                pos_prefixes = self.derivation_data['noun']['prefixes'] + self.derivation_data['compound_prefixes']
                pos2stress = self.derivation_data['noun']['noun2stress']

            if pos_prefixes:
                for prefix in pos_prefixes:
                    if word.startswith(prefix) and word[len(prefix):] in pos2stress:
                        stressed_forms = pos2stress[word[len(prefix):]]
                        stressed_form1 = list(stressed_forms)[0]  # это огрубление модели, так как могут быть случаи с 2 вариантами ударения.
                        stressed_word = prefix + stressed_form1
                        n_vowels = 0
                        for c in stressed_word:
                            if c in 'уеыаоэёяию':
                                n_vowels += 1
                            elif c == '\u0301':
                                return n_vowels

        # Есть продуктивные приставки типа НЕ
        for prefix in 'не'.split():
            if word.startswith(prefix):
                word1 = word[len(prefix):]  # отсекаем приставку
                if len(word1) > 2:  # нас интересуют составные слова
                    if word1 in self.word_accents_dict:
                        return self.get_vowel_count(prefix, abbrevs=False) + self.word_accents_dict[word1]

        # Иногда можно взять ударение из стема: "ПОЗИТРОННЫЙ" -> "ПОЗИТРОН"
        if False:
            stem = self.stemmer.stem(word)
            if stem in self.word_accents_dict:
                return self.word_accents_dict[stem]

        if vowel_count == 0:
            # знаки препинания и т.д., в которых нет ни одной гласной.
            return -1

        # 02.08.2022 Исправление опечатки - твердый знак вместо мягкого "пъянки"
        if 'ъ' in word:
            word1 = word.replace('ъ', 'ь')
            if word1 in self.word_accents_dict:
                return self.word_accents_dict[word1]

        if self.allow_nonrussian_accentuation is False and self.re_cyrlocator.search(word) is None:
            # Английские слова и т.д. не ударяем
            return 0

        if True:
            return self.predict_stress(word)

        return (vowel_count + 1) // 2

    def get_accents(self, word0, ud_tags):
        word = self.sanitize_word(word0)
        word = self.yoficate2(word, ud_tags)

        vowel_count = self.get_vowel_count(word)
        if vowel_count == 0:
            if len(word) == 1:
                # единственная согласная
                return [WordAccentuation.build_nostress()]
            elif word.lower() == 'ль':
                return [WordAccentuation.build_nostress()]

        if vowel_count == 1:
            # Для слов, содержащих единственную гласную, сразу возвращаем позицию ударения на этой гласной
            return [WordAccentuation.build_stress1()]

        secondary_accentuation = self.get_secondary_accentuation(word)

        if word in self.ambiguous_accents2:
            res = []
            for stress_pos in self.ambiguous_accents2[word]:
                res.append(WordAccentuation(stress_pos, secondary_accentuation))
            return res

        if word in self.word_accents_dict:
            stress_pos = self.word_accents_dict[word]
            return [WordAccentuation(stress_pos, secondary_accentuation)]

        if ud_tags and self.ambiguous_accents and word in self.ambiguous_accents:
            stress_pos = self.predict_ambiguous_accent(word, ud_tags)
            if stress_pos == -1:
                # POS Tagger не смог выбрать верный вариант тэгсета. Поэтому перебираем все варианты ударения.
                res = []
                for stressed_form in self.ambiguous_accents[word]:
                    n_vowels = 0
                    for c in stressed_form:
                        if c.lower() in 'уеыаоэёяию':
                            n_vowels += 1
                            if c.isupper():
                                stress_pos = n_vowels
                                res.append(WordAccentuation(stress_pos, secondary_accentuation))

                return res
            else:
                return [WordAccentuation(stress_pos, secondary_accentuation)]

        if 'ё' in word:
            if word in self.word_accents_dict:
                stress_pos = self.word_accents_dict[word]
                return [WordAccentuation(stress_pos, secondary_accentuation)]

            # считаем, что "ё" всегда ударная (исключение - слово "ёфикация" и однокоренные)
            n_vowels = 0
            for c in word:
                if c in 'уеыаоэёяию':
                    n_vowels += 1
                    if c == 'ё':
                        stress_pos = n_vowels
                        return [WordAccentuation(stress_pos, secondary_accentuation)]

        # 19.05.2022 в порошках и т.п. по законам жанра допускаются намеренные ошибки типа "ошыбка".
        # Попробуем скорректировать такие ошибки.
        corrections = [('тьса', 'тся'),  # рвЕтьса
                       ('тьса', 'ться'),  # скрытьса
                       ('ться', 'тся'),
                       ('юцца', 'ются'),
                       ('цца', 'ться'),
                       ('юца', 'ются'),  # бьюца
                       ('шы', 'ши'), ('жы', 'жи'), ('цы', 'ци'), ('щю', 'щу'), ('чю', 'чу'),
                       ('ща', 'сча'),
                       ('щя', 'ща'),  # щями
                       ("чя", "ча"),  # чящя
                       ("жэ", "же"),  # художэственный
                       ('цэ', 'це'), ('жо', 'жё'), ('шо', 'шё'), ('чо', 'чё'), ('чьк', 'чк'),
                       ('що', 'щё'),  # вощоный
                       ('щьк', 'щк'),
                       ('цк', 'тск'),  # 07.06.2022 децкий ==> детский
                       ('цца', 'тся'),  # 04.08.2022 "льюцца"
                       ('ъе', 'ьё'),  # бъется
                       ('ье', 'ъе'),  # сьЕли
                       ('сн', 'стн'),  # грусный
                       ('цц', 'тц'),  # браццы
                       ('цц', 'дц'),  # триццать
                       ('чт', 'чьт'),  # прячте
                       ('тьн', 'тн'),  # плОтьник
                       ('зд', 'сд'),  # здачу
                       ('тса', 'тся'),  # гнУтса
                       ]

        for m2 in corrections:
            if m2[0] in word:
                word2 = word.replace(m2[0], m2[1])
                if word2 in self.word_accents_dict:
                    stress_pos = self.word_accents_dict[word2]
                    return [WordAccentuation(stress_pos, secondary_accentuation)]

        # восстанавливаем мягкий знак в "стоиш" "сможеш"  "сбереч"
        # встретимса
        #        ^^^
        e_corrections = [('иш', 'ишь'),  # стоиш
                         ('еш', 'ешь'),  # сможеш
                         ('еч', 'ечь'),  # сбереч
                         ('мса', 'мся'),  # встретимса
                         ]
        for e1, e2 in e_corrections:
            if word.endswith(e1):
                word2 = word[:-len(e1)] + e2
                if word2 in self.word_accents_dict:
                    stress_pos = self.word_accents_dict[word2]
                    return [WordAccentuation(stress_pos, secondary_accentuation)]

        # убираем финальный "ь" после шипящих:
        # клавишь
        if re.search(r'[чшщ]ь$', word):
            word2 = word[:-1]
            if word2 in self.word_accents_dict:
                stress_pos = self.word_accents_dict[word2]
                return [WordAccentuation(stress_pos, secondary_accentuation)]

        # повтор согласных и гласных сокращаем до одной согласной:
        # щщупать
        if len(word) > 1:
            cn = re.search(r'(\w)\1', word, flags=re.I)
            if cn:
                c1 = cn.group(1)[0]
                try:
                    word2 = re.sub(c1+'{2,}', c1, word, flags=re.I)
                    if word2 in self.word_accents_dict:
                        stress_pos = self.word_accents_dict[word2]
                        return [WordAccentuation(stress_pos, secondary_accentuation)]
                except re.error:
                    print(traceback.format_exc())
                    exit(1)

        # Некоторые грамматические формы в русском языке имеют
        # фиксированное ударение.
        pos1 = word.find('ейш') # сильнейший, наимудрейшие
        if pos1 != -1:
            stress_pos = self.get_vowel_count(word[:pos1], abbrevs=False) + 1
            return [WordAccentuation(stress_pos, secondary_accentuation)]

        # TODO: !!! префиксальную деривация перевести на использование TRIE для подбора префикса !!!
        if ud_tags and 'VERB' in ud_tags:
            # Глагольная префиксальная деривация.
            for prefix in self.derivation_data['verb']['prefixes']:
                if word.startswith(prefix) and word[len(prefix):] in self.derivation_data['verb']['verb2stress']:
                    stressed_forms = self.derivation_data['verb']['verb2stress'][word[len(prefix):]]
                    res = []
                    for stressed_form in stressed_forms:
                        stressed_word = prefix + stressed_form
                        n_vowels = 0
                        for c in stressed_word:
                            if c in 'уеыаоэёяию':
                                n_vowels += 1
                            elif c == '\u0301':
                                stress_pos = n_vowels
                                res.append(WordAccentuation(stress_pos, secondary_accentuation))
                    return res

        elif ud_tags and any((pos in ud_tags) for pos in ('NOUN', 'PROPN', 'ADJ')):
            # Префиксальная деривация для существительных и прилагательных
            pos_prefixes = None
            pos2stress = None
            if 'ADJ' in ud_tags:
                pos_prefixes = self.derivation_data['adj']['prefixes'] + self.derivation_data['compound_prefixes']
                pos2stress = self.derivation_data['adj']['adj2stress']
            elif any((pos in ud_tags) for pos in ('NOUN', 'PROPN')):
                pos_prefixes = self.derivation_data['noun']['prefixes'] + self.derivation_data['compound_prefixes']
                pos2stress = self.derivation_data['noun']['noun2stress']

            if pos_prefixes:
                for prefix in pos_prefixes:
                    if word.startswith(prefix) and word[len(prefix):] in pos2stress:
                        stressed_forms = pos2stress[word[len(prefix):]]
                        res = []
                        for stressed_form in stressed_forms:
                            stressed_word = prefix + stressed_form

                            if not secondary_accentuation and prefix in self.derivation_data['compound2stress']:
                                prefix_stress = self.derivation_data['compound2stress'][prefix]
                                secondary_accentuation = []
                                n_vowels = 0
                                for c in prefix_stress:
                                    if c.lower() in 'уеыаоэёяию':
                                        n_vowels += 1
                                        if c in 'АЕЁИОУЫЭЮЯ':
                                            secondary_accentuation.append(2)
                                        else:
                                            secondary_accentuation.append(0)
                                for c in stressed_form.lower():
                                    if c in 'уеыаоэёяию':
                                        secondary_accentuation.append(0)

                            n_vowels = 0
                            for c in stressed_word:
                                if c in 'уеыаоэёяию':
                                    n_vowels += 1

                                elif c == '\u0301':
                                    stress_pos = n_vowels
                                    res.append(WordAccentuation(stress_pos, secondary_accentuation))
                        return res

        # Есть продуктивные приставки типа НЕ
        for prefix in 'не'.split():
            if word.startswith(prefix):
                word1 = word[len(prefix):]  # отсекаем приставку
                if len(word1) > 2:  # нас интересуют составные слова
                    if word1 in self.word_accents_dict:
                        stress_pos = self.get_vowel_count(prefix, abbrevs=False) + self.word_accents_dict[word1]
                        return [WordAccentuation(stress_pos, secondary_accentuation)]

        # Иногда можно взять ударение из стема: "ПОЗИТРОННЫЙ" -> "ПОЗИТРОН"
        if False:
            stem = self.stemmer.stem(word)
            if stem in self.word_accents_dict:
                return self.word_accents_dict[stem]

        if vowel_count == 0:
            # знаки препинания и т.д., в которых нет ни одной гласной.
            return [WordAccentuation.build_nostress()]

        # 02.08.2022 Исправление опечатки - твердый знак вместо мягкого "пъянки"
        if 'ъ' in word:
            word1 = word.replace('ъ', 'ь')
            if word1 in self.word_accents_dict:
                stress_pos = self.word_accents_dict[word1]
                return [WordAccentuation(stress_pos, secondary_accentuation)]

        if self.allow_nonrussian_accentuation is False and self.re_cyrlocator.search(word) is None:
            # Английские слова и т.д. не ударяем
            return [WordAccentuation.build_nostress()]

        stress_pos = self.predict_stress(word)
        return [WordAccentuation(stress_pos, None)]

    def get_phoneme(self, word):
        word = self.sanitize_word(word)

        word_end = word[-3:]
        vowel_count = self.get_vowel_count(word, abbrevs=False)
        accent = self.get_accent(word)

        return word_end, vowel_count, accent

    def render_accenture(self, word):
        accent = self.get_accent(word)

        accenture = []
        n_vowels = 0
        stress_found = False
        for c in word:
            s = None
            if c in 'уеыаоэёяию':
                n_vowels += 1
                s = '-'

            if n_vowels == accent and not stress_found:
                s = '^'
                stress_found = True

            if s:
                accenture.append(s)

        return ''.join(accenture)

    def do_endings_match(self, word1, vowels1, accent1, word2):
        if len(word1) >= 3 and len(word2) >= 3:
            # Если ударный последний слог, то проверим совпадение этого слога
            if accent1 == vowels1:
                # TODO - надо проверять не весь слог, а буквы, начиная с гласной
                # ...
                syllabs1 = rusyllab.split_word(word1)
                syllabs2 = rusyllab.split_word(word2)
                return syllabs1[-1] == syllabs2[-1]
            else:
                # В остальных случаях - проверим совместимость последних 3х букв
                end1 = word1[-3:]
                end2 = word2[-3:]

                # БЕДНА == ГРУСТНА
                if re.match(r'[бвгджзклмнпрстфхцчшщ]на', end1) and re.match(r'[бвгджзклмнпрстфхцчшщ]на', end2):
                    return True

                if re.match(r'[бвгджзклмнпрстфхцчшщ][ая]я', end1) and re.match(r'[бвгджзклмнпрстфхцчшщ][ая]я', end2):
                    return True

                if re.match(r'[бвгджзклмнпрстфхцчшщ][ую]ю', end1) and re.match(r'[бвгджзклмнпрстфхцчшщ][ую]ю', end2):
                    return True

                return end1 == end2

        return False


def get_stressed_vowel(word, stress):
    v_counter = 0
    for c in word:
        if c in "уеыаоэёяию":
            v_counter += 1
            if v_counter == stress:
                return c

    return None


def get_stressed_syllab(syllabs, stress):
    v_counter = 0
    for syllab in syllabs:
        for c in syllab:
            if c in "уеыаоэёяию":
                v_counter += 1
                if v_counter == stress:
                    return syllab

    return None


def are_rhymed_syllables(syllab1, syllab2):
    # Проверяем совпадение последних букв слога, начиная с гласной
    r1 = re.match(r'^.+([уеыаоэёяию].*)$', syllab1)
    r2 = re.match(r'^.+([уеыаоэёяию].*)$', syllab2)
    if r1 and r2:
        # это последние буквы слога с гласной.
        s1 = r1.group(1)
        s2 = r2.group(1)

        # при проверке соответствия надо учесть фонетическую совместимость гласных (vowel2base)
        return are_phonetically_equal(s1, s2)

    return False


def extract_ending_vc(s):
    # вернет последние буквы слова, среди которых минимум 1 гласная и 1 согласная

    e = {'твоего': 'во', 'моего': 'во'}.get(s)
    if e is not None:
        return e

    # мягкий знак и после него йотированная гласная:
    #
    # семья
    #    ^^
    if re.search(r'ь[ёеюя]$', s):
        return s[-1]

    # гласная и следом - йотированная гласная:
    #
    # моя
    #  ^^
    if re.search(r'[аеёиоуыэюя][ёеюя]$', s):
        return s[-1]

    # неглиже
    #      ^^
    r = re.search('([жшщ])е$', s)
    if r:
        return r.group(1) + 'э'

    # хороши
    #     ^^
    r = re.search('([жшщ])и$', s)
    if r:
        return r.group(1) + 'ы'

    # иногда встречается в пирожках неорфографичная форма:
    # щя
    # ^^
    r = re.search('([жшщ])я$', s)
    if r:
        return r.group(1) + 'а'

    # иногда встречается в пирожках неорфографичная форма:
    # трепещю
    #      ^^
    r = re.search('([жшщ])ю$', s)
    if r:
        return r.group(1) + 'у'

    # МАМА
    #   ^^
    r = re.search('([бвгджзйклмнпрстфхцчшщ][уеыаоэёяию]+)$', s)
    if r:
        return r.group(1)

    # СТОЛБ
    #   ^^^
    # СТОЙ
    #   ^^
    r = re.search('([уеыаоэёяию][бвгджзйклмнпрстфхцчшщ]+)$', s)
    if r:
        return r.group(1)

    # КРОВЬ
    #   ^^^
    r = re.search('([уеыаоэёяию][бвгджзйклмнпрстфхцчшщ]+ь)$', s)
    if r:
        return r.group(1)


    # ЛАДЬЯ
    #   ^^^
    r = re.search('([бвгджзйклмнпрстфхцчшщ]ь[уеыаоэёяию]+)$', s)
    if r:
        return r.group(1)

    return ''


vowel2base = {'я': 'а', 'ю': 'у', 'е': 'э'}
vowel2base0 = {'я': 'а', 'ю': 'у'}


def are_phonetically_equal(s1, s2):
    # Проверяем фонетическую эквивалентность двух строк, учитывая пары гласных типа А-Я etc
    # Каждая из строк содержит часть слова, начиная с ударной гласной (или с согласной перед ней).
    if len(s1) == len(s2):
        if s1 == s2:
            return True

        vowels = "уеыаоэёяию"
        total_vowvels1 = sum((c in vowels) for c in s1)

        n_vowel = 0
        for ic, (c1, c2) in enumerate(zip(s1, s2)):
            if c1 in vowels:
                n_vowel += 1
                if n_vowel == 1:
                    # УДАРНАЯ ГЛАСНАЯ
                    if total_vowvels1 == 1 and ic == len(s1)-1:
                        # ОТЕЛЯ <==> ДАЛА
                        if c1 != c2:
                            return False
                    else:
                        cc1 = vowel2base0.get(c1, c1)
                        cc2 = vowel2base0.get(c2, c2)
                        if cc1 != cc2:
                            return False

                        tail1 = s1[ic+1:]
                        tail2 = s2[ic+1:]
                        if tail1 in ('жной', 'жный', 'жнай') and tail2 in ('жной', 'жный', 'жнай'):
                            return True
                else:
                    cc1 = vowel2base.get(c1, c1)
                    cc2 = vowel2base.get(c2, c2)
                    if cc1 != cc2:
                        return False
            elif c1 != c2:
                return False

        return True

    return False


def transcript_unstressed(chars):
    if chars is None or len(chars) == 0:
        return ''

    phonems = []
    for c in chars:
        if c == 'о':
            phonems.append('а')
        elif c == 'и':
            phonems.append('ы')
        elif c == 'ю':
            phonems.append('у')
        elif c == 'я':
            phonems.append('а')
        elif c == 'ё':
            phonems.append('о')
        elif c == 'е':
            phonems.append('э')
        else:
            phonems.append(c)

    if phonems[-1] == 'ж':
        phonems[-1] = 'ш'
    if phonems[-1] == 'в':
        phonems[-1] = 'ф'
    elif phonems[-1] == 'б':
        # оглушение частицы "б"
        # не бу́дь у ба́нь и у кофе́ен
        # пиаротде́лов и прессслу́жб    <=====
        # мы все б завши́вели и пи́ли
        # из лу́ж б                    <=====
        phonems[-1] = 'п'

    res = ''.join(phonems)
    return res


def extract_ending_spelling_after_stress(accents, word, stress, ud_tags, unstressed_prefix, unstressed_tail):
    if len(word) == 1:
        return unstressed_prefix + word + unstressed_tail

    lword = word.lower()

    v_counter = 0
    ending = None
    for i, c in enumerate(lword): # 25.06.2022 приводим к нижнему регистру
        if c in "уеыаоэёяию":
            v_counter += 1
            if v_counter == stress:
                if i == len(word) - 1 and len(unstressed_tail) == 0:
                    # Ударная гласная в конце слова, берем последние 2 или 3 буквы
                    # ГУБА
                    #   ^^
                    ending = extract_ending_vc(word)
                else:
                    ending = word[i:]
                    if ud_tags is not None and ('ADJ' in ud_tags or 'DET' in ud_tags) and ending == 'ого':
                        # Меняем "люб-ОГО" на "люб-ОВО"
                        ending = 'ово'

                # if len(ending) < len(word):
                #     c2 = word[-len(ending)-1]
                #     if c2 in 'цшщ' and ending[0] == 'и':
                #         # меняем ЦИ -> ЦЫ
                #         ending = 'ы' + ending[1:]

                # if ending.endswith('ь'):  # убираем финальный мягкий знак: "ВОЗЬМЁШЬ"
                #     ending = ending[:-1]
                #
                # if ending.endswith('д'):  # оглушаем последнюю "д": ВЗГЛЯД
                #     ending = ending[:-1] + 'т'
                # elif ending.endswith('ж'):  # оглушаем последнюю "ж": ЁЖ
                #     ending = ending[:-1] + 'ш'
                # elif ending.endswith('з'):  # оглушаем последнюю "з": МОРОЗ
                #     ending = ending[:-1] + 'с'
                # #elif ending.endswith('г'):  # оглушаем последнюю "г": БОГ
                # #    ending = ending[:-1] + 'х'
                # elif ending.endswith('б'):  # оглушаем последнюю "б": ГРОБ
                #     ending = ending[:-1] + 'п'
                # elif ending.endswith('в'):  # оглушаем последнюю "в": КРОВ
                #     ending = ending[:-1] + 'ф'

                break

    if not ending:
        # print('ERROR@385 word1={} stress1={}'.format(word1, stress1))
        return ''

    if ending.startswith('ё'):
        ending = 'о' + ending[1:]

    return ending + unstressed_tail

def extract_ending_prononciation_after_stress(accents, word, stress, ud_tags, unstressed_prefix, unstressed_tail):
    unstressed_prefix_transcription = accents.pronounce(unstressed_prefix)  # transcript_unstressed(unstressed_prefix)
    unstressed_tail_transcription = accents.pronounce(unstressed_tail)  #transcript_unstressed(unstressed_tail)

    if len(word) == 1:
        return unstressed_prefix_transcription + word + unstressed_tail_transcription

    lword = word.lower()

    ending = {'его': 'во', 'него': 'во', 'сего': 'во', 'того': 'во', 'всякого': 'якава'}.get(lword)
    if ending is None:
        v_counter = 0
        for i, c in enumerate(lword): # 25.06.2022 приводим к нижнему регистру
            if c in "уеыаоэёяию":
                v_counter += 1
                if v_counter == stress:
                    if i == len(word) - 1 and len(unstressed_tail) == 0:
                        # Ударная гласная в конце слова, берем последние 2 или 3 буквы
                        # ГУБА
                        #   ^^
                        ending = extract_ending_vc(word)

                        # 01.02.2022 неударная "о" перед ударной гласной превращается в "а":  своя ==> сваЯ
                        if len(ending) >= 2 and ending[-2] == 'о' and ending[-1] in 'аеёиоуыэюя':
                            ending = ending[:-2] + 'а' + ending[-1]

                    else:
                        ending = word[i:]
                        if ud_tags is not None and ('ADJ' in ud_tags or 'DET' in ud_tags) and ending == 'ого':
                            # Меняем "люб-ОГО" на "люб-ОВО"
                            ending = 'ово'

                        if ending.startswith('е'):
                            # 01.02.2022 меняем ударную "е" на "э": летом==>л'Этом
                            ending = 'э' + ending[1:]
                        elif ending.startswith('я'):
                            # 01.02.2022 меняем ударную "я" на "а": мячик==>м'Ачик
                            ending = 'а' + ending[1:]
                        elif ending.startswith('ё'):
                            # 01.02.2022 меняем ударную "ё" на "о": мёдом==>м'Одом
                            ending = 'о' + ending[1:]
                        elif ending.startswith('ю'):
                            # 01.02.2022 меняем ударную "ю" на "у": люся==>л'Уся
                            ending = 'у' + ending[1:]
                        elif ending.startswith('и'):
                            # 01.02.2022 меняем ударную "и" на "ы": сливы==>сл'Ывы, живы==>жЫвы
                            ending = 'ы' + ending[1:]

                    if len(ending) < len(word):
                        c2 = word[-len(ending)-1]
                        if c2 in 'цшщ' and ending[0] == 'и':
                            # меняем ЦИ -> ЦЫ
                            ending = 'ы' + ending[1:]

                    # if ending.endswith('ь'):  # убираем финальный мягкий знак: "ВОЗЬМЁШЬ"
                    #     ending = ending[:-1]
                    #
                    # if ending.endswith('д'):  # оглушаем последнюю "д": ВЗГЛЯД
                    #     ending = ending[:-1] + 'т'
                    # elif ending.endswith('ж'):  # оглушаем последнюю "ж": ЁЖ
                    #     ending = ending[:-1] + 'ш'
                    # elif ending.endswith('з'):  # оглушаем последнюю "з": МОРОЗ
                    #     ending = ending[:-1] + 'с'
                    # #elif ending.endswith('г'):  # оглушаем последнюю "г": БОГ
                    # #    ending = ending[:-1] + 'х'
                    # elif ending.endswith('б'):  # оглушаем последнюю "б": ГРОБ
                    #     ending = ending[:-1] + 'п'
                    # elif ending.endswith('в'):  # оглушаем последнюю "в": КРОВ
                    #     ending = ending[:-1] + 'ф'

                    break

    if not ending:
        # print('ERROR@385 word1={} stress1={}'.format(word1, stress1))
        return ''

    ending = accents.pronounce(ending)
    if ending.startswith('ё'):
        ending = 'о' + ending[1:]

    return unstressed_prefix_transcription + ending + unstressed_tail_transcription


def rhymed(accents, word1, ud_tags1, word2, ud_tags2):
    word1 = accents.yoficate2(accents.sanitize_word(word1), ud_tags1)
    word2 = accents.yoficate2(accents.sanitize_word(word2), ud_tags2)

    if (word1.lower(), word2.lower()) in accents.rhymed_words or (word2.lower(), word1.lower()) in accents.rhymed_words:
        return True

    stress1 = accents.get_accent(word1, ud_tags1)
    vow_count1 = accents.get_vowel_count(word1)
    pos1 = vow_count1 - stress1

    stress2 = accents.get_accent(word2, ud_tags2)
    vow_count2 = accents.get_vowel_count(word2)
    pos2 = vow_count2 - stress2

    # смещение ударной гласной от конца слова должно быть одно и то же
    # для проверяемых слов.
    if pos1 == pos2:
        # 28.06.2022 особо рассматриваем случай рифмовки с местоимением "я": друзья-я
        if word2 == 'я':
            return word1.endswith('я')

        # Теперь все буквы, начиная с ударной гласной
        ending1 = extract_ending_prononciation_after_stress(accents, word1, stress1, ud_tags1, '', '')
        ending2 = extract_ending_prononciation_after_stress(accents, word2, stress2, ud_tags2, '', '')

        return are_phonetically_equal(ending1, ending2)

    return False


def rhymed2(accentuator, word1, stress1, ud_tags1, unstressed_prefix1, unstressed_tail1, word2, stress2, ud_tags2, unstressed_prefix2, unstressed_tail2):
    word1 = accentuator.yoficate2(accentuator.sanitize_word(word1), ud_tags1)
    word2 = accentuator.yoficate2(accentuator.sanitize_word(word2), ud_tags2)

    if not unstressed_tail1 and not unstressed_tail2:
        if (word1.lower(), word2.lower()) in accentuator.rhymed_words or (word2.lower(), word1.lower()) in accentuator.rhymed_words:
            return True

    vow_count1 = accentuator.get_vowel_count(word1)
    pos1 = vow_count1 - stress1 + accentuator.get_vowel_count(unstressed_tail1, abbrevs=False)

    vow_count2 = accentuator.get_vowel_count(word2)
    pos2 = vow_count2 - stress2 + accentuator.get_vowel_count(unstressed_tail2, abbrevs=False)

    # смещение ударной гласной от конца слова должно быть одно и то же
    # для проверяемых слов.
    if pos1 == pos2:
        # 22.05.2022 Особо рассматриваем рифмовку с местоимением "я":
        # пролета́ет ле́то
        # гру́сти не тая́
        # и аналоги́чно
        # пролета́ю я́
        if word2 == 'я' and len(word1) > 1 and word1[-2] in 'аеёиоуэюяь' and word1[-1] == 'я':
            return True

        # Детектирование рифмы "раб ли - грабли" осложнено оглушением согласной "б" в первом элементе.
        # Для простоты попробуем сначала сравнить буквальные заударные окончания элементов.
        ending1 = extract_ending_spelling_after_stress(accentuator, word1, stress1, ud_tags1, unstressed_prefix1, unstressed_tail1)
        ending2 = extract_ending_spelling_after_stress(accentuator, word2, stress2, ud_tags2, unstressed_prefix2, unstressed_tail2)
        if ending1 == ending2:
            return True

        # Получаем клаузулы - все буквы, начиная с ударной гласной
        ending1 = extract_ending_prononciation_after_stress(accentuator, word1, stress1, ud_tags1, unstressed_prefix1, unstressed_tail1)
        ending2 = extract_ending_prononciation_after_stress(accentuator, word2, stress2, ud_tags2, unstressed_prefix2, unstressed_tail2)

        # Фонетическое сравнение клаузул.
        return are_phonetically_equal(ending1, ending2)

    return False



fuzzy_ending_pairs0 = [
    #(r'\^эсна', r'\^эстна'), # интересно - честно
    (r'\^а([:C:])кэ', r'\^а[:1:]ках'),  # word1=стакашке xword1=стак^ашкэ clausula1=^ашкэ  word2=ромашках xword2=рам^ашках clausula2=^ашках
    (r'м\^ая', r'м\^ают'),  # word1=прямая xword1=прам^ая clausula1=^ая  word2=понимают xword2=паным^ают clausula2=^ают
    (r'ч\^асье', r'щ\^астья'),  # word1=одночасье xword1=аднач^асье clausula1=^асье  word2=счастья xword2=щ^астья clausula2=^астья
    (r'([:C:])\^амы', r'[:1:]\^аминь'),  # word1=руками xword1=рук^амы clausula1=^амы  word2=камень xword2=к^аминь clausula2=^аминь
    (r'\^эзыл[ыа]', r'\^эзы[юи]'),  # word1=отгрезили xword1=атгр^эзылы clausula1=^эзылы  word2=Поэзию xword2=па^эзыю clausula2=^эзыю
    (r'\^э([:C:])тый', r'\^э[:1:]тэ'),  # word1=скуфентий xword1=скуф^энтый clausula1=^энтый  word2=моменте xword2=мам^энтэ clausula2=^энтэ
    (r'\^о([:C:])ка', r'\^о[:1:]кай'),  # word1=сторонка xword1=стар^онка clausula1=^онка  word2=иконкой xword2=ик^онкай clausula2=^онкай
    (r'([:C:])\^овые', r'[:1:]\^овая'),  # word1=бордовые xword1=бард^овые clausula1=^овые  word2=бедовая xword2=бид^овая clausula2=^овая
    (r'([:C:])к\^у', r'[:1:]к\^у[фптмрс]'),  # тоску - скуф
    (r'[:C:]\^эса', r'[:C:]\^эза'),  # word1=леса xword1=л^эса clausula1=^эса  word2=железо xword2=жил^эза clausula2=^эза
    (r'\^один', r'\^одын'),  # xword1=биспл^один xword2=^одын
    (r'\^асный', r'\^азны'),  # xword1=ап^асный xword2=сабл^азны  word1=опасный word2=соблазны
    (r'\^ошый', r'\^ожый'),  # word1=хороший xword1=хар^ошый clausula1=^ошый  word2=толстокожий xword2=талстак^ожый clausula2=^ожый
    (r'\^астна', r'\^асна'),  # word1=страстно xword1=стр^астна clausula1=^астна  word2=согласна xword2=сагл^асна clausula2=^асна
    (r'\^очкай', r'\^очка'),  # word1=бочкой xword1=б^очкай clausula1=^очкай  word2=дочка xword2=д^очка clausula2=^очка
    (r'[:C:]\^ам', r'[:C:]\^ам'),  # word1=волнам xword1=валн^ам clausula1=^ам  word2=там xword2=т^ам clausula2=^ам
    (r'\^ыпкаю', r'\^ыпкае'),  # word1=рыбкою xword1=р^ыпкаю clausula1=^ыпкаю  word2=зыбкое xword2=з^ыпкае clausula2=^ыпкае
    (r'\^енье', r'\^энья'),  # word1=настроенье xword1=настра^енье clausula1=^енье  word2=сомненья xword2=самн^энья clausula2=^энья
    (r'\^о([:C:])кай', r'\^о[:1:]ку'),  # word1=походкой xword1=пах^откай clausula1=^откай  word2=водку xword2=в^отку clausula2=^отку
    (r'ъ\^ела', r'\^элам'),  # word1=съела xword1=съ^ела clausula1=^ела  word2=делом xword2=д^элам clausula2=^элам
    (r'([:C:])\^ус', r'[:1:]\^урс'),  # word1=вкус xword1=фк^ус clausula1=^ус  word2=курс xword2=к^урс clausula2=^урс
    (r'а\^ю', r'а\^ют'),  # word1=краю xword1=кра^ю xword2=па^ют
    (r'([:C:])\^анья', r'[:1:]\^аньем'),  # word1=прощанья xword1=пращ^анья clausula1=^анья  word2=обещаньем xword2=абищ^аньем clausula2=^аньем
    (r'\^ашн([:A:])', r'\^ажн[:1:]'),  # word1=бесшабашно xword1=бисшаб^ашна clausula1=^ашна  word2=важно xword2=в^ажна clausula2=^ажна
    (r'\^янствам', r'\^анства'),  # word1=постоянством xword1=паста^янствам clausula1=^янствам  word2=убранства xword2=убр^анства clausula2=^анства
    (r'\^ыстка', r'\^ыска'),  # word1=эвритмистка xword1=эврытм^ыстка clausula1=^ыстка  word2=василиска xword2=васыл^ыска clausula2=^ыска
    (r'\^аньтэ', r'\^янты'),  # word1=перестаньте xword1=пирист^аньтэ clausula1=^аньтэ  word2=дамаянти xword2=дама^янты clausula2=^янты
    (r'\^э([:C:])на', r'\^э[:1:]най'),  # word1=напевно word2=повседневной
    (r'([:C:])\^ает', r'[:1:]\^аять'),  # word1=хватает xword1=хват^ает clausula1=^ает  word2=таять xword2=т^аять clausula2=^аять
    (r'\^эздый', r'\^ездыт'),  # word1=созвездий xword1=сазв^эздый clausula1=^эздый  word2=объездит xword2=абъ^ездыт clausula2=^ездыт
    (r'\^ылась', r'\^ыласть'),  # word1=стремилась xword1=стрим^ылась clausula1=^ылась  word2=немилость xword2=ним^ыласть clausula2=^ыласть
    (r'\^эздны', r'\^эзный'),  # word1=бездны xword1=б^эздны clausula1=^эздны  word2=болезный xword2=бал^эзный clausula2=^эзный
    (r'а\^я([:C:])', r'\^а[:1:]'),  # word1=хаям xword1=ха^ям clausula1=^ям  word2=там xword2=т^ам clausula2=^ам
    (r'\^эжн([:A:])', r'\^эшн[:1:]'),  # word1=нежно xword1=н^эжна clausula1=^эжна  word2=поспешно xword2=пасп^эшна clausula2=^эшна
    (r'\^э([:C:])([:C:])([:A:])', r'\^е[:1:][:2:][:3:]'),  # word1=клетки xword1=кл^эткы clausula1=^эткы  word2=объедки xword2=абъ^еткы clausula2=^еткы
    (r'\^ычный', r'\^ычна'),  # word1=столичный xword1=стал^ычный clausula1=^ычный  word2=неприлично xword2=нипрыл^ычна clausula2=^ычна
    (r'\^я([:C:])ый', r'\^а[:1:]ы'),  # word1=объятий xword1=абъ^ятый clausula1=^ятый  word2=благодати xword2=благад^аты clausula2=^аты
    (r'\^ужных', r'\^ужна'),  # word1=жемчужных xword1=жимч^ужных clausula1=^ужных  word2=безоружно xword2=бизар^ужна clausula2=^ужна
    (r'\^([:A:])чна', r'\^[:1:]чнай'),  # word1=публично xword1=публ^ычна clausula1=^ычна  word2=ироничной xword2=иран^ычнай clausula2=^ычнай
    (r'ав\^э', r'аф\^э'),  # word1=траве xword1=трав^э clausula1=в^э  word2=строфе xword2=страф^э clausula2=ф^э
    (r'л\^эю', r'л\^эя'),  # word1=аллею xword1=ал^эю clausula1=^эю  word2=сожалея xword2=сажал^эя clausula2=^эя
    (r'\^эснай', r'\^эзнай'),  # word1=небесной xword1=ниб^эснай clausula1=^эснай  word2=железной xword2=жил^эзнай clausula2=^эзнай
    (r'\^э([:C:])([:C:])ы', r'\^э[:1:][:2:]а'),  # word1=дверцы xword1=дв^эрцы clausula1=^эрцы  word2=сердца xword2=с^эрца clausula2=^эрца
    (r'([:C:])\^([:A:])ршн', r'[:1:]\^[:2:]шн'),  # xword1=дыст^оршн xword2=дат^ошн  word1=дисторшн word2=дотошн
    (r'([:C:])\^([:A:])тск', r'[:1:]\^[:2:]цк'),  # xword1=бр^атск xword2=дур^ацк  word1=братск word2=дурацк
    (r'([:C:])\^([:A:])ль', r'[:1:]\^[:2:]льть'),  # xword1=кыс^эль xword2=с^эльть  word1=кисель word2=сельдь
    (r'\^([:A:])шынай', r'\^[:1:]шинай'),  # xword1=м^ашынай xword2=нинакр^ашинай  word1=машиной word2=ненакрашенной
    (r'\^([:A:])чик', r'\^[:1:]чирк'),  # xword1=п^очик xword2=^очирк  word1=почек word2=очерк
    (r'([:C:])\^([:A:])ч', r'[:1:]\^[:2:]тч'),  # xword1=пл^ач xword2=кл^атч  word1=плач word2=клатч
    (r'\^([:A:])сила', r'\^[:1:]сыла'),  # xword1=в^эсила xword2=пав^эсыла  word1=весело word2=повесила
    (r'([:C:])\^([:A:])', r'[:1:]\^[:2:]нт'),  # word1=небеса xword1=нибис^а clausula1=с^а  word2=десант xword2=дис^ант clausula2=^ант
    (r'([:C:])ь\^([:A:])', r'[:1:]ь\^[:2:]т'),  # xword1=барталамь^ю xword2=мь^ют  word1=барталамью word2=мьют
    (r'\^([:A:])й([:C:])ь', r'\^[:1:][:2:]ь'),  # xword1=какт^эйль xword2=к^эль  word1=коктейль word2=келль
    (r'([:C:])\^([:A:])м', r'[:1:]\^[:2:]н'),  # xword1=д^ум xword2=бад^ун  word1=дум word2=бодун
    (r'([:C:])\^([:A:])л', r'[:1:]\^[:2:]м'),  # xword1=паш^ол xword2=кавш^ом  word1=пошёл word2=ковшом
    (r'([:C:])\^([:A:])', r'[:1:]\^[:2:]йт'),  # xword1=афс^а xword2=афс^айт  word1=овса word2=офсайд
    (r'([:C:])\^([:A:])т', r'[:1:]\^[:2:]нт'),  # xword1=сик^ут xword2=сик^унт  word1=секут word2=секунд
    (r'([:C:])\^([:A:])', r'[:1:]\^[:2:]знь'),  # xword1=наж^ы xword2=ж^ызнь  word1=ножи word2=жизнь
    (r'([:C:])\^([:A:])н', r'[:1:]\^[:2:]м'),  # xword1=ад^ын xword2=вад^ым  word1=один word2=вадим
    (r'([:C:])\^([:A:])т', r'[:1:]\^[:2:]рт'),  # xword1=т^от xword2=нит^орт  word1=тот word2=неторт
    (r'([:C:])\^([:A:])й', r'[:1:]\^[:2:]ль'),  # xword1=гаст^эй xword2=паст^эль  word1=гостей word2=постель
    (r'([:C:])\^([:A:])м', r'[:1:]\^[:2:]рм'),  # xword1=плавнык^ом xword2=к^орм  word1=плавником word2=корм
    (r'([:C:])\^([:A:])ть', r'[:1:]\^[:2:]сть'),  # xword1=с^эть xword2=с^эсть  word1=сеть word2=сесть
    (r'([:C:])\^([:A:])т', r'[:1:]\^[:2:]кт'),  # xword1=эстр^ат xword2=экстр^акт  word1=эстрад word2=экстракт
    (r'([:C:])\^([:A:])ф', r'[:1:]\^[:2:]т'),  # xword1=вилыч^аф xword2=урч^ат  word1=величав word2=урчат
    (r'\^ышкам', r'\^ышкы'),  # xword1=сл^ышкам xword2=кн^ышкы  word1=слишком word2=книжки
    (r'\^ая', r'\^айа'),  # немая - майя
    (r'\^эзнам', r'\^эсна'),  # полезном - неизвестно
    (r'\^эем', r'\^эю'),  # xword1=шалф^эем xword2=усп^эю  word1=шалфеем word2=успею
    (r'\^атасть', r'\^адасть'),  # xword1=чрив^атасть xword2=мл^адасть  word1=чреватость word2=младость
    (r'д\^ы', r'т\^ы'),  # xword1=сад^ы xword2=т^ы  word1=сады word2=ты
    (r'\^еньи', r'\^энье'),  # xword1=настра^еньи xword2=правыд^энье  word1=настроеньи word2=провиденье

    (r'([:C:])\^([:A:])([:C:])ь', r'[:1:]\^[:2:][:3:]'),  # # xword1=вр^ось xword2=нивр^ос  word1=врозь word2=невроз
    (r'([:C:])\^([:A:])', r'[:1:]\^[:2:][:C:]ь'),  # xword1=еж^а xword2=ж^аль  word1=ежа word2=жаль
    (r'([:C:])\^([:A:])([:C:])', r'[:1:]\^[:2:][:3:]ь'),  # xword1=дж^ус xword2=стыж^усь  word1=джус word2=стыжусь
    (r'([:C:])\^([:A:])ть', r'[:1:]\^[:2:]'),  # xword1=б^ыть xword2=судьб^ы  word1=быть word2=судьбы
    (r'([:C:])\^([:A:])', r'[:1:]\^[:2:][:C:]'),  # xword1=тайг^а xword2=г^ат  word1=тайга word2=гад
    (r'([:C:])\^([:A:])([:C:])', r'[:1:]\^[:2:]'),  # xword1=тик^ут xword2=кук^у  word1=текут word2=куку
    (r'[чщ]\^([:A:])', r'[чщ]\^[:1:]'),  # xword1=стуч^а xword2=барщ^а  word1=стуча word2=борща
    (r'\^([:A:])([:C:])[чшщ]', r'\^[:1:][:2:][чшщ]'),  # xword1=п^орш xword2=б^орщ  word1=порш word2=борщ
    (r'([:A:])\^ы[:C:]', r'[:1:]\^ы'),  # xword1=ста^ыт xword2=сва^ы  word1=стоит word2=свои
    (r'\^е([:C:])([:C:])', r'\^э[:1:][:2:]'),  # xword1=пра^ест xword2=^эст  word1=проезд word2=ест
    (r'\^([:A:])([:C:])[:C:]ь', r'\^[:1:][:2:]ь'),  # xword1=ч^ысть xword2=уч^ысь  word1=чисть word2=учись
    (r'([:C:])\^([:A:])[:C:]([:C:])', r'[:1:]\^[:2:][:3:]'),  # xword1=дыкт^ант xword2=дыкт^ат  word1=диктант word2=диктат
    (r'([:A:])\^([:A:])[:C:]', r'[:1:]\^[:2:]'),  # xword1=на^ук xword2=а^у  word1=наук word2=ау
    (r'([:C:])\^([:A:])', r'[:1:]\^[:2:]й'),  # xword1=граш^а xword2=лыш^ай  word1=гроша word2=лишай
    (r'\^([:A:])снь', r'\^[:1:]знь'),  # xword1=п^эснь xword2=бал^эзнь  word1=песнь word2=болезнь
    (r'([:C:])\^([:A:])й', r'[:1:]\^[:2:]ть'),  # xword1=ч^ай xword2=нач^ать  word1=чай word2=начать
    (r'([:C:])\^([:A:])([:C:])([:C:])', r'[:1:]\^[:2:][:C:][:3:][:4:]'),  # xword1=т^эст xword2=т^экст  word1=тест word2=текст
    (r'\^э([:C:])([:C:])', r'[:A:]\^е[:1:][:2:]'),  # xword1=м^эст xword2=у^ест  word1=мест word2=уезд
    (r'т\^([:A:])', r'д\^[:1:]'),  # xword1=накт^э xword2=гд^э  word1=ногте word2=где
    (r'([:C:])\^([:A:])ль', r'[:1:]\^[:2:]й'),  # xword1=б^оль xword2=таб^ой  word1=боль word2=тобой
    (r'([:C:])\^а', r'[:1:]ь\^я'),  # xword1=звин^а xword2=свынь^я  word1=звеня word2=свинья
    (r'ц\^([:A:])', r'тс\^[:1:]'),  # xword1=кальц^о xword2=лытс^о  word1=кольцо word2=литсо
    (r'ж\^([:A:])', r'ш\^[:1:]'),  # xword1=уж^э xword2=душ^э  word1=уже word2=душе
    (r'п\^([:A:])', r'б\^[:1:]'),  # xword1=сып^а xword2=сиб^а  word1=сипя word2=себя
    (r'\^([:A:])сть', r'\^[:1:]зть'),  # xword1=наиз^усть xword2=гр^узть  word1=наизусть word2=груздь
    (r'ш\^([:A:])', r'ж\^[:1:]'),  # xword1=душ^э xword2=уж^э  word1=душе word2=уже
    (r'\^([:A:])зть', r'\^[:1:]сть'),  # xword1=гр^узть xword2=гр^усть  word1=груздь word2=грусть
    (r'([:C:])\^([:A:])йс', r'[:1:]\^[:2:]льс'),  # xword1=р^эйс xword2=р^эльс  word1=рейс word2=рельс
    (r'\^([:A:])([:C:])([:C:])', r'\^[:1:][:2:][:3:]ь'),  # xword1=п^уст xword2=п^усть  word1=пуст word2=пусть
    (r'([:C:])\^([:A:])й', r'[:1:]\^[:2:]'),  # xword1=бурж^уй xword2=еж^у  word1=буржуй word2=ежу
    (r'([:C:])\^([:A:])([:C:])[:C:]([:C:])', r'[:1:]\^[:2:][:3:][:4:]'),  # xword1=б^эздн xword2=луб^эзн  word1=бездн word2=любезн
    (r'\^([:A:])([:C:])', r'\^[:1:][:2:]ь'),  # xword1=с^ыр xword2=паз^ырь  word1=сыр word2=позырь
    (r'\^([:A:])[:C:]([:C:])([:C:])ь', r'\^[:1:][:2:][:3:]ь'),  # xword1=анс^амбль xword2=с^абль  word1=ансамбль word2=сабль
    (r'([:C:])ь\^ё', r'[:1:]ь\^ё[:C:]'),  # xword1=жнывь^ё xword2=жывь^ём  word1=жнивьё word2=живьём
    (r'\^([:A:])сь', r'\^[:1:]сть'),  # xword1=атрад^ась xword2=матч^асть  word1=отродясь word2=матчасть
    (r'([:C:])\^([:A:])сь', r'[:1:]\^[:2:]'),  # xword1=нист^ысь xword2=нист^ы  word1=нестись word2=нести

    (r'ь\^ян', r'ь\^ям'),  # xword1=пь^ян xword2=друзь^ям  word1=пьян word2=друзьям
    (r'[:C:]\^ач', r'[:A:]\^яч'),  # xword1=м^ач xword2=ху^яч  word1=мяч word2=хуячь
    (r'[бвгджзклмнпрстфхцчшщл]ь\^ер', r'\^е'),  # модельер - ателье
    (r'д\^ы', r'д\^ы[:C:]'),  # xword1=фпирид^ы xword2=крид^ыт  word1=впереди word2=кредит
    (r'в\^ы[:C:]', r'в\^ы'),  # xword1=в^ыт xword2=лав^ы  word1=вид word2=лови
    (r'р\^а[:C:]', r'р\^а[:C:]'),  # xword1=жыр^аф xword2=жыр^ах  word1=жираф word2=жирах
    (r'\^ысх', r'\^ыск'),  # xword1=в^ысх xword2=франц^ыск  word1=визг word2=франциск
    (r'\^удл', r'\^утл'),  # xword1=д^удл xword2=ад^утл  word1=дудл word2=одутл

    (r'ш\^от', r'ш\^о'),  # word1=шот xword1=ш^от clausula1=^от  word2=хорошо xword2=хараш^о clausula2=ш^о

    (r'ч\^о[:C:]', r'ч\^о'),  # xword1=ч^ом xword2=плич^о  word1=чом word2=плечо
    (r'з\^а', r'з\^а[:C:]ь'),  # xword1=глаз^а xword2=каз^ань  word1=глаза word2=казань
    (r'с\^а', r'с\^а[:C:]'),  # xword1=прынис^а xword2=ис^ак  word1=принеся word2=иссяк
    (r'т\^а', r'т\^а[:C:]'),  # xword1=тиснат^а xword2=т^ам  word1=теснота word2=там
    (r'н\^а', r'н\^а[:C:]ь'),  # xword1=мин^а xword2=мин^ать  word1=меня word2=менять
    (r'ж\^ал', r'ж\^аль'),  # xword1=лиж^ал xword2=ж^аль  word1=лежал word2=жаль
    (r'р\^оть', r'р\^от'),  # xword1=спар^оть xword2=бар^от  word1=спороть word2=борот
    (r'х\^о[:C:]', r'х\^о'),  # xword1=мх^ом xword2=имх^о  word1=мхом word2=имхо
    (r'ц\^э[:C:]т', r'ц\^э[:C:]т'),  # xword1=фармац^эфт xword2=риц^эпт  word1=фармацевт word2=рецепт
    (r'л\^о[:C:]', r'л\^о[:C:]'),  # xword1=пл^ох xword2=сл^оф  word1=плох word2=слов
    (r'т\^орх', r'т\^ок'),  # xword1=васт^орх xword2=васт^ок  word1=восторг word2=восток
    (r'ст\^а[:C:]', r'ст\^а[:C:]'),  # xword1=двухст^ах xword2=ст^аф  word1=двухстах word2=став
    (r'в\^эс[:C:]ь', r'в\^эсь'),  # xword1=в^эсть xword2=в^эсь  word1=весть word2=весь
    (r'ч\^о[:C:]т', r'ч\^от'),  # xword1=ч^орт xword2=тич^от  word1=чорт word2=течот
    (r'с\^э[:C:]', r'с\^э'),  # xword1=аблыс^эл xword2=фс^э  word1=облысел word2=все
    (r'л\^а[:C:]', r'л\^а'),  # xword1=сл^ап xword2=прышл^а  word1=слаб word2=пришла
    (r'л\^э', r'л\^э[:C:]'),  # xword1=бажал^э xword2=жал^эл  word1=божоле word2=жалел
    (r'[:A:]\^ест', r'\^эсть'),  # xword1=пра^ест xword2=^эсть  word1=проезд word2=есть
    (r'\^ун', r'[:A:]\^юнь'),  # xword1=^ун xword2=и^юнь  word1=юн word2=июнь
    (r'т\^эп', r'т\^эпь'),  # xword1=ст^эп xword2=ст^эпь  word1=степ word2=степь
    (r'к\^ат', r'к\^ать'),  # xword1=канфыск^ат xword2=иск^ать  word1=конфискат word2=искать
    (r'р\^а', r'р\^а[:C:]'),  # xword1=актабр^а xword2=бабр^ат  word1=октября word2=бобрят
    (r'т\^о[:C:]', r'т\^о[:C:]'),  # xword1=минт^оф xword2=минт^ол  word1=ментов word2=ментол

    (r'в\^ан', r'в\^ам'),  #

    (r'ж\^ы', r'ж\^ыть'), # word1=бомжи xword1=бамж^ы clausula1=ж^ы  word2=жить xword2=ж^ыть clausula2=^ыть

    (r'ж\^эй', r'ж\^э'), # word1=бомжей xword1=бамж^эй clausula1=^эй  word2=уже xword2=уж^э clausula2=ж^э

    (r'ц\^ом', r'ц\^о'),  # word1=кольцом xword1=кальц^ом clausula1=^ом  word2=лицо xword2=лыц^о clausula2=ц^о

    (r'\^осх', r'\^оск'),  # word1=мозг xword1=м^осх clausula1=^осх  word2=киоск xword2=кы^оск clausula2=^оск

    (r'п\^ох', r'п\^о'),  # word1=пох xword1=п^ох clausula1=^ох  word2=по xword2=п^о clausula2=п^о

    (r'р\^оф', r'р\^о'),  # word1=ветров xword1=витр^оф clausula1=^оф  word2=метро xword2=митр^о clausula2=р^о

    (r'\^оскы', r'\^осткый'),  # word1=берёзки xword1=бир^оскы clausula1=^оскы  word2=громоздкий xword2=грам^осткый clausula2=^осткый

    (r'\^эрты', r'\^эрьтэ'),  # смерти - верьте

    (r'б\^а', r'б\^ат'),  # word1=себя xword1=сиб^а clausula1=б^а  word2=ребят xword2=риб^ат clausula2=^ат

    (r'\^очим', r'\^очинь'),  # word1=впрочем xword1=фпр^очим clausula1=^очим  word2=очень xword2=^очинь clausula2=^очинь

    (r'\^опатам', r'\^опытам'),  # шёпотом - опытом

    (r'т\^у', r'д\^у'),  # лету - упаду

    (r'\^осы', r'\^озы'),  # косы - морозы

    (r'т\^эм', r'т\^э'),  # тем - полноте

    (r'\^эрыть', r'\^эры'),  # верить - двери

    (r'а\^я[бвгджзклмнпрстфхцчшщ]', r'а\^я'),  # краях - моя

    (r'\^оф', r'\^офь'),  # миров - вновь

    (r'\^этир', r'\^эчир'),  # ветер - вечер

    (r'\^эныя', r'\^эные'),  # творения - дуновение

    (r'\^опалим', r'\^опалэ'),  # тополем - во поле

    (r'исл\^а', r'исл\^ась'),  # несла - неслась

    (r'\^эба', r'\^эмбы'),  # небо - всем бы

    (r'\^эзнам', r'\^эстна'),  # полезном - неизвестно

    (r'\^он', r'\^ом'),  # балкон - знакОм

    (r'м\^ать', r'м\^а'),  # разломать - закрома

    (r'\^ожнасть', r'\^ожна'),  # word1=возможность clausula1=^ожнасть  word2=сложно clausula2=^ожна

    (r'\^ушать', r'\^ушу'),  # word1=слушать clausula1=^ушать  word2=душу clausula2=^ушу

    (r'\^атиль', r'\^атэ'),  # word1=издатель clausula1=^атиль  word2=дате clausula2=^атэ

    (r'х\^а', r'к\^а'),  # пастуха - издалека

    (r'\^осинь', r'\^осынь'),  # осень - просинь

    (r'\^ют', r'\^ут'),  # встают - орут

    (r'\^ож', r'\^ош'),  # дрожь - вернёшь

    (r'\^ёт', r'\^от'),  # пьёт - вперёд

    (r'\^олк', r'\^олх'), # толк - долг

    (r'\^афскэ', r'\^аскэ'),  # Заславске - сказке

    (r'\^аца', r'\^адцать'),  # заняться - восемнадцать

    (r'\^эсний', r'\^эсны'),  # интересней - песни

    (r'\^ыцы', r'\^ыца'), # птицы - разбиться

    (r'\^уца', r'\^удцэ'),  # льются - блюдце

    (r'л\^э', r'л\^эт'),  # нуле - амулет

    (r'\^уж[уыэа]', r'\^уж[уыэа]'),  # стужу - лужи

    (r'\^утам', r'\^уты'),  # парашютам - тьфу ты

    (r'\^ады', r'\^аты'),  # пощады - борща ты

    (r'\^овый', r'\^овай'),  # трёхочковый - волочковой

    (r'\^ымым', r'\^ымам'),  # одерж^имым - ж^имом

    (r'\^ыськы', r'\^ыскы'),  # с^иськи - в^иски

    (r'\^альцэ', r'\^альца'),  # еб^альце - п^альца

    (r'\^озин', r'\^озэ'),  # вирту^озен - п^озе

    (r'\^убым', r'\^убам'),  # грубым - ледорубом

    (r'\^ыскра', r'\^ыстра'),  # и́скра - кани́стра

    (r'\^ызар', r'\^ыза'),  # телевизор - антифриза

    (r'\^анай', r'\^аный'),  # манной - странный

    (r'\^очна', r'\^очный'),  # нарочно - молочный

    (r'\^юц[эа]', r'\^удц[аэ]'),  # льются - блюдце

    (r'[:C:]\^ое', r'[:C:]\^оя'),  # простое - покоя

    (r'\^ает', r'\^ают'),  # чает - повенчают

    (r'\^ывый', r'\^ыва'),  # белогривый - некрасиво

    (r'\^энья', r'\^энье'),  # настроенья - упоенье

    (r'\^айна', r'\^айнай'),  # неслучайно - тайной

    (r'\^овым', r'\^овам'),  # еловым - основам

    (r'\^авай', r'\^ава'),  #  славой - права

    (r'\^([:A:][:C:]+)а', r'\^([:A:][:C:]+)э'),  # риска - миске

    (r'\^ыны', r'\^ына'),  # цепеллины - господина

    (r'\^([:A:][:C:][:A:][:C:])а', r'\^([:A:][:C:][:A:][:C:])ам'), # яруса - парусом

    (r'\^ысил', r'\^ысыл'),  # чисел - возвысил

    (r'\^([:A:][:C:]{2,})э', r'\^([:A:][:C:]{2,})ай'),  # миске - пропиской

    (r'\^([:A:][щч])ие', r'\^([:A:][щч])ые'),  # щемящее - парящие

    (r'([:C:])\^ою', r'([:C:])\^ое'),  # щекою - такое

    (r'\^([:A:][:C:])ай', r'\^([:A:][:C:])а'),  # прохладой - надо

    (r'\^([:A:][жш])[эау]', r'\^([:A:][жш])[эау]'),  # коже - тревожа

    (r'\^([:A:]ш)ин', r'\^([:A:]ш)ан'),  #  вишен - услышан

    (r'\^([:A:][:C:]+ь)э', r'\^([:A:][:C:]+ь)а'),  # спасенье - вознесенья

    (r'\^([:A:][:C:]+)[ыэ]', r'\^([:A:][:C:]+)[ыэ]'),  # медведи - велосипеде

    (r'\^([:A:][:C:]+)[оуа]', r'\^([:A:][:C:]+)[оау]м'),  # Андрюшка - хрюшкам

    (r'\^([:A:][:C:]+[ыэ])[й]', r'\^([:A:][:C:]+[ыэ])'),  # первый-нервы

    (r'([:C:]+\^[ыэ])[й]', r'([:C:]+\^[ыэ])'),  # свиней-войне

    (r'\^([:A:][:C:]+)[оуа][мн]', r'\^([:A:][:C:]+)[оау]'),  # сонетом - Света

    (r'([:C:])\^[оуа]н', r'([:C:])\^[оау]'),  # Антон - манто

    (r'(\^[:A:][:C:]+)[оуа]', r'(\^[:A:][:C:]+)[оау]'),  # ложа - кожу

    (r'\^арт', r'\^аркт'),  # word1=март xword1=м^арт clausula1=^арт  word2=инфаркт xword2=инф^аркт clausula2=^аркт
    (r'р\^ам', r'р\^а'),  # word1=срам xword1=ср^ам clausula1=^ам  word2=ура xword2=ур^а clausula2=р^а
    (r'ц\^э', r'с\^э'),  # word1=лице xword1=лыц^э clausula1=ц^э  word2=эссе xword2=эсс^э clausula2=с^э
    (r'сн\^у', r'сн\^ул'),  # word1=весну xword1=висн^у clausula1=н^у  word2=уснул xword2=усн^ул clausula2=^ул
    (r'ч\^ы', r'ч\^ыт'),  # word1=мочи xword1=мач^ы clausula1=ч^ы  word2=кричит xword2=крыч^ыт clausula2=^ыт
    (r'\^оцк', r'\^отск'),  # word1=клёцк xword1=кл^оцк clausula1=^оцк  word2=пофлотск xword2=пафл^отск clausula2=^отск
    (r'ч\^уп', r'ч\^у'),  # word1=чуб xword1=ч^уп clausula1=^уп  word2=хочу xword2=хач^у clausula2=ч^у
    (r'з\^уп', r'з\^у'),  # word1=зуб xword1=з^уп clausula1=^уп  word2=грызу xword2=грыз^у clausula2=з^у
    (r'к\^ы', r'г\^ы'),  # word1=носки xword1=наск^ы clausula1=к^ы  word2=мозги xword2=мазг^ы clausula2=г^ы
    (r'ав\^у', r'а\^у'),  # word1=траву xword1=трав^у clausula1=в^у  word2=ау xword2=а^у clausula2=а^у
    (r'сь\^е', r'съ\^ел'),  # word1=месье xword1=мись^е clausula1=сь^е  word2=съел xword2=съ^ел clausula2=^ел
    (r'рт\^уть', r'рт\^у'),  # word1=ртуть xword1=рт^уть clausula1=^уть  word2=рту xword2=рт^у clausula2=т^у
    (r'аст\^ак', r'аст\^ах'),  # word1=костяк xword1=каст^ак clausula1=^ак  word2=гостях xword2=гаст^ах clausula2=^ах
    (r'ж\^ыф', r'ж\^ы'),  # word1=жив xword1=ж^ыф clausula1=^ыф  word2=скажи xword2=скаж^ы clausula2=ж^ы
    (r'ч\^а', r'ч\^ай'),  # word1=свеча xword1=свич^а clausula1=ч^а  word2=венчай xword2=винч^ай clausula2=^ай
    (r'ч\^а', r'ч\^ать'),  # word1=свеча xword1=свич^а clausula1=ч^а  word2=начать xword2=нач^ать clausula2=^ать
    (r'ус\^ок', r'ус\^ох'),  # word1=кусок xword1=кус^ок clausula1=^ок  word2=усох xword2=ус^ох clausula2=^ох
    (r'ш\^у', r'ш\^уй'),  # word1=прошу xword1=праш^у clausula1=ш^у  word2=феншуй xword2=финш^уй clausula2=^уй
    (r'сн\^э', r'сн\^эх'),  # word1=сне xword1=сн^э clausula1=н^э  word2=снег xword2=сн^эх clausula2=^эх
    (r'л\^эсть', r'л\^эсь'),  # word1=лесть xword1=л^эсть clausula1=^эсть  word2=лезь xword2=л^эсь clausula2=^эсь
    (r'ан\^ах', r'ан\^ак'),  # word1=монах xword1=ман^ах clausula1=^ах  word2=монак xword2=ман^ак clausula2=^ак
    (r'т\^ы', r'т\^ыт'),  # word1=пути xword1=пут^ы clausula1=т^ы  word2=летит xword2=лит^ыт clausula2=^ыт
    (r'т\^о', r'т\^ом'),  # word1=то xword1=т^о clausula1=т^о  word2=потом xword2=пат^ом clausula2=^ом
    (r'щ\^ы', r'щ\^ын'),  # word1=борщи xword1=барщ^ы clausula1=щ^ы  word2=мущщин xword2=мущ^ын clausula2=^ын
    (r'м\^ох', r'м\^орх'),  # word1=мог xword1=м^ох clausula1=^ох  word2=морг xword2=м^орх clausula2=^орх
    (r'ж\^ызнь', r'ж\^ысь'),  # word1=жизнь xword1=ж^ызнь clausula1=^ызнь  word2=держись xword2=дирж^ысь clausula2=^ысь
    (r'ск\^ы', r'ск\^ыт'),  # word1=мазки xword1=маск^ы clausula1=к^ы  word2=москит xword2=маск^ыт clausula2=^ыт
    (r'ч\^а', r'ч\^ах'),  # word1=ильича xword1=ильич^а clausula1=ч^а  word2=лучах xword2=луч^ах clausula2=^ах
    (r'са\^юс', r'су\^юсь'),  # word1=евросоюз xword1=евраса^юс clausula1=^юс  word2=суюсь xword2=су^юсь clausula2=^юсь
    (r'а\^юсь', r'а\^юс'),  # word1=боюсь xword1=ба^юсь clausula1=^юсь  word2=союз xword2=са^юс clausula2=^юс
    (r'щ\^а', r'щ\^ай'),  # word1=плаща xword1=плащ^а clausula1=щ^а  word2=прощай xword2=пращ^ай clausula2=^ай
    (r'\^ость', r'\^ось'),  # word1=кость xword1=к^ость clausula1=^ость  word2=пришлось xword2=прышл^ось clausula2=^ось
    (r'\^уств', r'\^устф'),  # word1=чувств xword1=ч^уств clausula1=^уств  word2=искусств xword2=иск^устф clausula2=^устф
    (r'д\^эль', r'д\^э'),  # word1=модель xword1=мад^эль clausula1=^эль  word2=дэ xword2=д^э clausula2=д^э
    (r'в\^ыч', r'ль\^ыч'),  # word1=свитч xword1=св^ыч clausula1=в^ыч  word2=ильич xword2=иль^ыч clausula2=^ыч
    (r'ст\^ах', r'ст\^а'),  # word1=местах xword1=мист^ах clausula1=^ах  word2=ста xword2=ст^а clausula2=т^а
    (r'ал\^а', r'ал\^ат'),  # word1=стола xword1=стал^а clausula1=л^а  word2=салат xword2=сал^ат clausula2=^ат
    (r'ачк\^ом', r'ачк\^о'),  # word1=бочком xword1=бачк^ом clausula1=^ом  word2=очко xword2=ачк^о clausula2=к^о
    (r'в\^эк', r'в\^э'),  # word1=век xword1=в^эк clausula1=^эк  word2=две xword2=дв^э clausula2=в^э
    (r'ф\^экт', r'ф\^эт'),  # word1=эффект xword1=эф^экт clausula1=^экт  word2=фет xword2=ф^эт clausula2=^эт
    (r'р\^эш', r'р\^эзж'),  # word1=брешь xword1=бр^эш clausula1=^эш  word2=брезжь xword2=бр^эзж clausula2=^эзж
    (r'ч\^ум', r'ч\^ун'),  # word1=чум xword1=ч^ум clausula1=^ум  word2=молчун xword2=малч^ун clausula2=^ун
    (r'п\^алм', r'п\^альм'),  # word1=напалм xword1=нап^алм clausula1=^алм  word2=пальм xword2=п^альм clausula2=^альм
    (r'\^озть', r'\^ость'),  # word1=гвоздь xword1=гв^озть clausula1=^озть  word2=гость xword2=г^ость clausula2=^ость
    (r'н\^ы', r'н\^ык'),  # word1=возни xword1=вазн^ы clausula1=н^ы  word2=ник xword2=н^ык clausula2=^ык
    (r'г\^ы', r'г\^ыл'),  # word1=враги xword1=враг^ы clausula1=г^ы  word2=могил xword2=маг^ыл clausula2=^ыл
    (r'\^яф', r'\^афь'),  # word1=объяв xword1=абъ^яф clausula1=^яф  word2=явь xword2=^афь clausula2=^афь
    (r'б\^арт', r'б\^ат'),  # word1=бард xword1=б^арт clausula1=^арт  word2=бад xword2=б^ат clausula2=^ат
    (r'ш\^ыт', r'ш\^ыть'),  # word1=спешит xword1=спиш^ыт clausula1=^ыт  word2=пришить xword2=прыш^ыть clausula2=^ыть
    (r'н\^у', r'н\^ул'),  # word1=жену xword1=жин^у clausula1=н^у  word2=вздрочнул xword2=вздрачн^ул clausula2=^ул
    (r'м\^ым', r'м\^ын'),  # word1=мим xword1=м^ым clausula1=^ым  word2=камин xword2=кам^ын clausula2=^ын
    (r'к\^ах', r'к\^а'),  # word1=руках xword1=рук^ах clausula1=^ах  word2=слегка xword2=сликк^а clausula2=к^а
    (r'п\^ы', r'п\^ыл'),  # word1=цепи xword1=цип^ы clausula1=п^ы  word2=цепил xword2=цип^ыл clausula2=^ыл
    (r'ж\^ык', r'ж\^ы'),  # word1=мужик xword1=муж^ык clausula1=^ык  word2=держи xword2=дирж^ы clausula2=ж^ы
    (r'ст\^а', r'ст\^ань'),  # word1=хвоста xword1=хваст^а clausula1=т^а  word2=отстань xword2=атст^ань clausula2=^ань
    (r'х\^ал', r'х\^а'),  # word1=бухал xword1=бух^ал clausula1=^ал  word2=хаха xword2=хах^а clausula2=х^а
    (r'ш\^ы', r'ш\^ым'),  # word1=карандаши xword1=карандаш^ы clausula1=ш^ы  word2=шуршим xword2=шурш^ым clausula2=^ым
    (r'ш\^ым', r'ш\^ы'),  # word1=пошуршим xword1=пашурш^ым clausula1=^ым  word2=души xword2=душ^ы clausula2=ш^ы
    (r'з\^ат', r'з\^ак'),  # word1=зад xword1=з^ат clausula1=^ат  word2=бальзак xword2=бальз^ак clausula2=^ак
    (r'г\^ып', r'г\^ы'),  # word1=изгиб xword1=изг^ып clausula1=^ып  word2=мозги xword2=мазг^ы clausula2=г^ы
    (r'к\^альп', r'к\^айп'),  # word1=скальп xword1=ск^альп clausula1=^альп  word2=скайп xword2=ск^айп clausula2=^айп
    (r'ш\^ум', r'ш\^у'),  # word1=шум xword1=ш^ум clausula1=^ум  word2=вишу xword2=выш^у clausula2=ш^у
    (r'д\^а', r'д\^ак'),  # word1=еда xword1=ед^а clausula1=д^а  word2=мудак xword2=муд^ак clausula2=^ак
    (r'ц\^э', r'ц\^эпт'),  # word1=конце xword1=канц^э clausula1=ц^э  word2=концепт xword2=канц^эпт clausula2=^эпт
    (r'л\^ыск', r'л\^ыськ'),  # word1=василиск xword1=васыл^ыск clausula1=^ыск  word2=вселись xword2=фсил^ыськ clausula2=^ыськ
    (r'н\^ах', r'н\^а'),  # word1=похоронах xword1=пахаран^ах clausula1=^ах  word2=она xword2=ан^а clausula2=н^а
    (r'ш\^о', r'ш\^ок'),  # word1=хорошо xword1=хараш^о clausula1=ш^о  word2=кишок xword2=кыш^ок clausula2=^ок
    (r'з\^у', r'з\^ут'),  # word1=козу xword1=каз^у clausula1=з^у  word2=грызут xword2=грыз^ут clausula2=^ут
    (r'б\^уть', r'б\^у'),  # word1=нибудь xword1=ныб^уть clausula1=^уть  word2=гробу xword2=граб^у clausula2=б^у
    (r'с\^ых', r'с\^ы'),  # word1=косых xword1=кас^ых clausula1=^ых  word2=спаси xword2=спас^ы clausula2=с^ы
    (r'к\^от', r'к\^о'),  # word1=антрикот xword1=антрык^от clausula1=^от  word2=трико xword2=трык^о clausula2=к^о
    (r'н\^ас', r'н\^ась'),  # word1=нас xword1=н^ас clausula1=^ас  word2=дразнясь xword2=дразн^ась clausula2=^ась
    (r'г\^ы', r'г\^ыт'),  # word1=яги xword1=яг^ы clausula1=г^ы  word2=гид xword2=г^ыт clausula2=^ыт
    (r'л\^ын', r'л\^ым'),  # word1=исполин xword1=испал^ын clausula1=^ын  word2=спалим xword2=спал^ым clausula2=^ым
    (r'ск\^ы', r'ск\^ым'),  # word1=волоски xword1=валаск^ы clausula1=к^ы  word2=морским xword2=марск^ым clausula2=^ым
    (r'сь\^е', r'съ\^ем'),  # word1=досье xword1=дась^е clausula1=сь^е  word2=досъем xword2=дасъ^ем clausula2=^ем
    (r'б\^у', r'б\^укф'),  # word1=избу xword1=изб^у clausula1=б^у  word2=букв xword2=б^укф clausula2=^укф
    (r'уж\^э', r'уш\^эй'),  # word1=уже xword1=уж^э clausula1=ж^э  word2=ушей xword2=уш^эй clausula2=^эй
    (r'т\^у', r'т\^ут'),  # word1=тату xword1=тат^у clausula1=т^у  word2=тут xword2=т^ут clausula2=^ут
    (r'н\^о', r'н\^ом'),  # word1=темно xword1=тимн^о clausula1=н^о  word2=вином xword2=вын^ом clausula2=^ом
    (r'тр\^ы', r'тр\^ым'),  # word1=три xword1=тр^ы clausula1=р^ы  word2=экстрим xword2=экстр^ым clausula2=^ым
    (r'л\^ос', r'л\^ось'),  # word1=волос xword1=вал^ос clausula1=^ос  word2=спалось xword2=спал^ось clausula2=^ось
    (r'г\^ын', r'г\^ымн'),  # word1=вагин xword1=ваг^ын clausula1=^ын  word2=гимн xword2=г^ымн clausula2=^ымн
    (r'л\^ыст', r'л\^ысь'),  # word1=лист xword1=л^ыст clausula1=^ыст  word2=сдались xword2=сдал^ысь clausula2=^ысь
    (r'с\^э', r'с\^эм'),  # word1=лисе xword1=лыс^э clausula1=с^э  word2=совсем xword2=сафс^эм clausula2=^эм
    (r'щ\^ом', r'щ\^о'),  # word1=борщом xword1=барщ^ом clausula1=^ом  word2=ещё xword2=ещ^о clausula2=щ^о
    (r'\^этч', r'\^эч'),  # word1=кимбербетч xword1=кымбирб^этч clausula1=^этч  word2=привлечь xword2=прывл^эч clausula2=^эч
    (r'ц\^оф', r'ц\^о'),  # word1=огурцов xword1=агурц^оф clausula1=^оф  word2=яйцо xword2=яйц^о clausula2=ц^о
    (r'х\^от', r'х\^о'),  # word1=обход xword1=апх^от clausula1=^от  word2=имхо xword2=имх^о clausula2=х^о
    (r'с\^эмь', r'с\^эм'),  # word1=семь xword1=с^эмь clausula1=^эмь  word2=совсем xword2=сафс^эм clausula2=^эм
    (r'с\^эмь', r'па\^ем'),  # word1=семь xword1=с^эмь clausula1=^эмь  word2=поем xword2=па^ем clausula2=^ем
    (r'з\^ыф', r'з\^ы'),  # word1=позыв xword1=паз^ыф clausula1=^ыф  word2=пазы xword2=паз^ы clausula2=з^ы
    (r'с\^а', r'с\^ал'),  # word1=пса xword1=пс^а clausula1=с^а  word2=проссал xword2=прас^ал clausula2=^ал
    (r'б\^оф', r'б\^о'),  # word1=жлобов xword1=жлаб^оф clausula1=^оф  word2=жабо xword2=жаб^о clausula2=б^о
    (r'л\^оф', r'л\^о'),  # word1=плов xword1=пл^оф clausula1=^оф  word2=ебло xword2=ебл^о clausula2=л^о
    (r'щ\^эй', r'щ\^э'),  # word1=вещей xword1=вищ^эй clausula1=^эй  word2=борще xword2=барщ^э clausula2=щ^э
    (r'с\^ы', r'с\^ыр'),  # word1=усы xword1=ус^ы clausula1=с^ы  word2=сыр xword2=с^ыр clausula2=^ыр
    (r'ч\^ал', r'ч\^а'),  # word1=постучал xword1=пастуч^ал clausula1=^ал  word2=моча xword2=мач^а clausula2=ч^а
    (r'т\^ы', r'т\^ых'),  # word1=кресты xword1=крист^ы clausula1=т^ы  word2=пустых xword2=пуст^ых clausula2=^ых
    (r'г\^ы', r'г\^ых'),  # word1=круги xword1=круг^ы clausula1=г^ы  word2=других xword2=друг^ых clausula2=^ых
    (r'мн\^э', r'мн\^эл'),  # word1=мне xword1=мн^э clausula1=н^э  word2=стемнел xword2=стимн^эл clausula2=^эл
    (r'н\^ыльс', r'н\^ылс'),  # word1=нильс xword1=н^ыльс clausula1=^ыльс  word2=снилс xword2=сн^ылс clausula2=^ылс
    (r'зд\^ок', r'сд\^ох'),  # word1=ездок xword1=езд^ок clausula1=^ок  word2=сдох xword2=сд^ох clausula2=^ох
    (r'г\^ул', r'г\^у'),  # word1=гул xword1=г^ул clausula1=^ул  word2=бегу xword2=биг^у clausula2=г^у
    (r'с\^ын', r'с\^ым'),  # word1=сын xword1=с^ын clausula1=^ын  word2=ссым xword2=с^ым clausula2=^ым
    (r'д\^ым', r'д\^ы'),  # word1=молодым xword1=малад^ым clausula1=^ым  word2=воды xword2=вад^ы clausula2=д^ы
    (r'\^юс', r'\^ус'),  # word1=боюсссс xword1=ба^юс clausula1=^юс  word2=флюс xword2=фл^ус clausula2=^ус
    (r'ст\^о', r'ст\^ол'),  # word1=сто xword1=ст^о clausula1=т^о  word2=стол xword2=ст^ол clausula2=^ол
    (r'ж\^ы', r'ж\^ысь'),  # word1=ржи xword1=рж^ы clausula1=ж^ы  word2=держись xword2=дирж^ысь clausula2=^ысь
    (r'к\^ы', r'к\^ылт'),  # word1=портки xword1=партк^ы clausula1=к^ы  word2=килт xword2=к^ылт clausula2=^ылт
    (r'з\^а', r'з\^ам'),  # word1=глаза xword1=глаз^а clausula1=з^а  word2=слезам xword2=слиз^ам clausula2=^ам
    (r'\^эсь', r'\^эс'),  # word1=весь xword1=в^эсь clausula1=^эсь  word2=воскрес xword2=васкр^эс clausula2=^эс
    (r'м\^эрть', r'м\^эть'),  # word1=смерть xword1=см^эрть clausula1=^эрть  word2=сметь xword2=см^эть clausula2=^эть
    (r'ш\^ать', r'ш\^а'),  # word1=дышать xword1=дыш^ать clausula1=^ать  word2=душа xword2=душ^а clausula2=ш^а
    (r'ч\^ам', r'ч\^а'),  # word1=врачам xword1=врач^ам clausula1=^ам  word2=моча xword2=мач^а clausula2=ч^а
    (r'ж\^ыт', r'ж\^ы'),  # word1=дрожит xword1=драж^ыт clausula1=^ыт  word2=джи xword2=дж^ы clausula2=ж^ы
    (r'ч\^асть', r'ч\^ась'),  # word1=часть xword1=ч^асть clausula1=^асть  word2=мочась xword2=мач^ась clausula2=^ась
    (r'\^ысь', r'\^ыс'),  # word1=окстись xword1=акст^ысь clausula1=^ысь  word2=абсцисс xword2=апсц^ыс clausula2=^ыс
    (r'с\^ыл', r'с\^ы'),  # word1=попросил xword1=папрас^ыл clausula1=^ыл  word2=такси xword2=такс^ы clausula2=с^ы
    (r'т\^ус', r'т\^усь'),  # word1=блютус xword1=блут^ус clausula1=^ус  word2=плетусь xword2=плит^усь clausula2=^усь
    (r'дь\^ю', r'д\^у'),  # word1=дью xword1=дь^ю clausula1=дь^ю  word2=фондю xword2=фанд^у clausula2=д^у
    (r'ж\^ар', r'ж\^а'),  # word1=пожар xword1=паж^ар clausula1=^ар  word2=пажа xword2=паж^а clausula2=ж^а
    (r'\^эц', r'\^эцъ'),  # word1=конец xword1=кан^эц clausula1=^эц  word2=пиздецъ xword2=пызд^эцъ clausula2=^эцъ
    (r'ф\^эр', r'ф\^э'),  # word1=фер xword1=ф^эр clausula1=^эр  word2=кафе xword2=каф^э clausula2=ф^э
    (r'з\^у', r'з\^ум'),  # word1=стезю xword1=стиз^у clausula1=з^у  word2=изюм xword2=из^ум clausula2=^ум
    (r'ц\^ам', r'ц\^а'),  # word1=огурцам xword1=агурц^ам clausula1=^ам  word2=юцца xword2=юц^а clausula2=ц^а
    (r'т\^ок', r'т\^о'),  # word1=ток xword1=т^ок clausula1=^ок  word2=никто xword2=ныкт^о clausula2=т^о
    (r'ш\^ы', r'ш\^ыт'),  # word1=души xword1=душ^ы clausula1=ш^ы  word2=кишит xword2=кыш^ыт clausula2=^ыт
    (r'з\^ат', r'з\^а'),  # word1=назад xword1=наз^ат clausula1=^ат  word2=глаза xword2=глаз^а clausula2=з^а
    (r'ж\^а', r'ж\^ап'),  # word1=ежа xword1=еж^а clausula1=ж^а  word2=жаб xword2=ж^ап clausula2=^ап
    (r'з\^ал', r'з\^а'),  # word1=взял xword1=вз^ал clausula1=^ал  word2=нельзя xword2=нильз^а clausula2=з^а
    (r'ч\^ок', r'ч\^о'),  # word1=маячок xword1=маяч^ок clausula1=^ок  word2=чо xword2=ч^о clausula2=ч^о
    (r'х\^ы', r'х\^ых'),  # word1=ухи xword1=ух^ы clausula1=х^ы  word2=бухих xword2=бух^ых clausula2=^ых
    (r'ж\^ым', r'ж\^ын'),  # word1=зажим xword1=заж^ым clausula1=^ым  word2=джинн xword2=дж^ын clausula2=^ын
    (r'ш\^ол', r'ш\^о'),  # word1=пришол xword1=прыш^ол clausula1=^ол  word2=ышшо xword2=ыш^о clausula2=ш^о
    (r'к\^ак', r'к\^а'),  # word1=никак xword1=нык^ак clausula1=^ак  word2=штыка xword2=штык^а clausula2=к^а
    (r'д\^ы', r'д\^ыть'),  # word1=поди xword1=пад^ы clausula1=д^ы  word2=бродить xword2=брад^ыть clausula2=^ыть
    (r'б\^ы', r'б\^ыл'),  # word1=губы xword1=губ^ы clausula1=б^ы  word2=был xword2=б^ыл clausula2=^ыл
    (r'ж\^ай', r'щ\^ай'),  # уезжай прощай
]

def compile_rhyme_rx(s):
    for x, y in [(':C:', 'бвгджзклмнпрстфхцчшщт'), (':A:', 'аоеёиуыюэюя')]:
        s = s.replace(x, y)

    rx = re.compile(s + '$')
    return rx


def preprocess_rhyme_rx(s):
    for x, y in [(':C:', 'бвгджзклмнпрстфхцчшщт'), (':A:', 'аоеёиуыюэюя')]:
        s = s.replace(x, y)

    return s



def count_ref_groups(rx_text):
    return len(re.findall(r'\[:\d:\]', rx_text))


fuzzy_ending_pairs = [(compile_rhyme_rx(s1), compile_rhyme_rx(s2), s1, s2, count_ref_groups(s2)) for s1, s2 in fuzzy_ending_pairs0]


def check_ending_rx_matching_2(word1, word2, r1, r2):
    m1 = r1.search(word1)
    if m1:
        m2 = r2.search(word2)
        if m2:
            for g1, g2 in zip(m1.groups(), m2.groups()):
                if g1 != g2:
                    return False

            return True

    return False


def check_ending_rx_matching_3(word1, word2, r1, r2, rx_text1, rx_text2, num_groups):
    m1 = r1.search(word1)
    if m1:
        if num_groups == 4:
            rx_text22 = rx_text2.replace('[:1:]', m1.group(1)).replace('[:2:]', m1.group(2)).replace('[:3:]', m1.group(3)).replace('[:4:]', m1.group(4))
        elif num_groups == 3:
            rx_text22 = rx_text2.replace('[:1:]', m1.group(1)).replace('[:2:]', m1.group(2)).replace('[:3:]', m1.group(3))
        elif num_groups == 2:
            rx_text22 = rx_text2.replace('[:1:]', m1.group(1)).replace('[:2:]', m1.group(2))
        elif num_groups == 1:
            rx_text22 = rx_text2.replace('[:1:]', m1.group(1))
        else:
            raise NotImplementedError()

        m2 = re.search(preprocess_rhyme_rx(rx_text22) + '$', word2)
        if m2:
            for g1, g2 in zip(m1.groups(), m2.groups()):
                if g1 != g2:
                    return False

            return True

    return False



xword_cases = {
('чувств', 1): ('ч^уств', '^уств'),
('чувства', 1): ('ч^уства', '^уства'),
('чувством', 1): ('ч^уствам', '^уствам'),
('чувству', 1): ('ч^уству', '^уству'),
('чувстве', 1): ('ч^устве', '^устве'),

('скотч', 1): ('ск^оч', '^оч'),
('скотче', 1): ('ск^оче', '^оче'),
('скотчем', 1): ('ск^очем', '^очем'),
('скотчу', 1): ('ск^очу', '^очу'),
('скотча', 1): ('ск^оча', '^оча'),

('эссе', 2): ('эсс^э', 'с^э'),

('бездну', 1): ('б^эзну', 'б^эзну'),

    ('свитч', 1): ('св^ыч', 'в^ыч'),
}

def render_xword(accentuator, word, stress_pos, ud_tags, unstressed_prefix, unstressed_tail):
    k = (word, stress_pos)
    if k in xword_cases:
        return xword_cases[k]

    if k == ('сердца', 1):
        # xword, clausula
        xword_cases[k] = ('с^эрца', '^эрца')
        return xword_cases[k]

    unstressed_prefix_transcript = transcript_unstressed(unstressed_prefix)
    unstressed_tail_transcript = transcript_unstressed(unstressed_tail)

    phonems = []

    VOWELS = 'уеыаоэёяию'

    # 07-12-2022 ситуации с последним БЕЗУДАРНЫМ словом "я":
    # так э́то де́лать ви́д что я́
    #                    ^^^^^
    if word == 'я' and stress_pos == 1 and unstressed_prefix in VOWELS:
        return "^я", unstressed_prefix+'^я'

    # Упрощенный алгоритм фонетической транскрипции - не учитываем йотирование, для гласных июяеё не помечаем
    # смягчение предшествующих согласных, etc.

    # 06.02.2024 коррекция цепочек согласных в конце: БОЮСССС ==> БОЮС
    word = re.sub(r'([бвгджзклмнпрстфхцчшщъьй])\1+$', r'\1', word, flags=re.I)

    v_counter = 0
    for i, c in enumerate(word.lower()):
        if c in VOWELS:
            v_counter += 1
            if v_counter == stress_pos:
                # Достигли ударения
                # Вставляем символ "^"
                phonems.append('^')

                ending = word[i:]
                if ud_tags is not None and ('ADJ' in ud_tags or 'DET' in ud_tags) and ending == 'ого' and word not in ['отлого',]:
                    # Меняем "люб-ОГО" на "люб-ОВО"
                    phonems.extend('ова')
                    break
                elif ending[1:] in ('ться', 'тся'):
                    if c == 'е':
                        c = 'э'
                    elif c == 'я':
                        c = 'а'
                    elif c == 'ё':
                        c = 'о'
                    elif c == 'ю':
                        c = 'у'
                    elif c == 'и':
                        c = 'ы'

                    phonems.append(c)
                    phonems.append('ц')
                    phonems.extend('а')
                    break
                else:
                    # Добавляем ударную гласную и продолжаем обрабатывать символы справа от него как безударные
                    if i > 0 and word[i-1] in ('ьъ'+VOWELS) and c in 'еёюя':
                        # 07-12-2022 в будущем надо сделать йотирование, а пока в случае паттернов типа "ружья" оставляем
                        # гласные "е", "ё", "ю", "я"
                        pass
                    elif c == 'е':
                        c = 'э'
                    elif c == 'я':
                        c = 'а'
                    elif c == 'ё':
                        c = 'о'
                    elif c == 'ю':
                        c = 'у'
                    elif c == 'и':
                        # 01.02.2022 меняем ударную "и" на "ы": сливы==>сл'Ывы, живы==>жЫвы
                        c = 'ы'

                    phonems.append(c)
            else:
                # Еще не достигли ударения или находимся справа от него.
                ending = word[i:]

                if ud_tags is not None and ('ADJ' in ud_tags or 'DET' in ud_tags) and ending == 'ого':
                    # местного ==> мэстнава
                    phonems.extend('ава')
                    break

                if c == 'о':
                    # безударная "о" превращается в "а"
                    c = 'а'
                elif c == 'е':
                    if len(phonems) == 0 or phonems[-1] in VOWELS+'ь':
                        # первую в слове, и после гласной, 'е' оставляем (должно быть что-то типа je)
                        pass
                    else:
                        # металле ==> митал'э
                        if i == len(word)-1:
                            c = 'э'
                        else:
                            c = 'и'
                elif c == 'я':
                    if len(phonems) == 0 or phonems[-1] in VOWELS+'ь':
                        pass
                    else:
                        c = 'а'
                elif c == 'ё':
                    if len(phonems) == 0 or phonems[-1] in VOWELS:
                        pass
                    else:
                        c = 'о'
                elif c == 'ю':
                    if len(phonems) == 0 or phonems[-1] in VOWELS+'ь':
                        pass
                    else:
                        c = 'у'
                elif c == 'и':
                    if len(phonems) == 0 or phonems[-1] in VOWELS+'ь':
                        pass
                    else:
                        # меняем ЦИ -> ЦЫ
                        #if c2 in 'цшщ' and ending[0] == 'и':
                        c = 'ы'

                phonems.append(c)
        else:
            # строго говоря, согласные надо бы смягчать в зависимости от следующей буквы (еёиюяь).
            # но нам для разметки стихов это не нужно.

            if c == 'ж':
                # превращается в "ш", если дальше идет глухая согласная
                # прожка ==> прошка
                if i < len(word)-1 and word[i+1] in 'пфктс':
                    c = 'ш'

            if i == len(word)-1:
                if c == 'д':  # последняя "д" оглушается до "т":  ВЗГЛЯД
                    c = 'т'
                elif c == 'ж':  # оглушаем последнюю "ж": ЁЖ
                    c = 'ш'
                elif c == 'з':  # оглушаем последнюю "з": МОРОЗ
                    c = 'с'
                elif c == 'г':  # оглушаем последнюю "г": БОГ
                    c = 'х'
                elif c == 'б':  # оглушаем последнюю "б": ГРОБ
                    c = 'п'
                elif c == 'в':  # оглушаем последнюю "в": КРОВ
                    c = 'ф'

            phonems.append(c)

    if len(phonems) > 2 and phonems[-1] == 'ь' and phonems[-2] in 'шчжщ':  # убираем финальный мягкий знак: "ВОЗЬМЁШЬ", РОЖЬ, МЫШЬ, ДРОЖЬ
        phonems = phonems[:-1]

    xword = unstressed_prefix_transcript + ''.join(phonems) + unstressed_tail_transcript
    #xword = accentuator.pronounce(xword)

    # ЧЕСТНО ==> ЧЭСНА
    # ИЗВЕСТНО ==> ИЗВЭСНА
    # ИЗВЕСТНЫ ==> ИЗВЭСНЫ
    # ИЗВЕСТНАЯ ==> ИЗВЭСНАЯ
    xword = re.sub('эстн(а|ы|ый|ая|ае|ую|ых|ым|ымы|ым|ава|аму|ам)$', r'эсн\1', xword)

    # ГРОМОЗДКИЙ ==> ГРАМОСТКИЙ
    xword = xword.replace('здк', 'стк')

    # СОЛНЦЕ -> СОНЦЕ
    xword = xword.replace('лнц', 'нц')

    # СЧАСТЬЕ -> ЩАСТЬЕ
    xword = xword.replace('сч', 'щ')

    # БРАТЬСЯ -> БРАЦА
    xword = xword.replace('ться', 'ца')

    # БОЯТСЯ -> БОЯЦА
    xword = xword.replace('тся', 'ца')

    # БРАТЦЫ -> БРАЦЫ
    xword = xword.replace('тц', 'ц')

    #
    #         # ЖИР -> ЖЫР
    #         s = s.replace('жи', 'жы')
    #
    #         # ШИП -> ШЫП
    #         s = s.replace('ши', 'шы')
    #
    #         # МОЦИОН -> МОЦЫОН
    #         s = s.replace('ци', 'цы')
    #
    #         # ЖЁСТКО -> ЖОСТКО
    #         s = s.replace('жё', 'жо')
    #
    #         # ОКОНЦЕ -> ОКОНЦЭ
    #         s = s.replace('це', 'цэ')
    #

    # двойные согласные:
    # СУББОТА -> СУБОТА
    xword = re.sub(r'([бвгджзклмнпрстфхцчшщ])\1', r'\1', xword)

    # оглушение:
    # СКОБКУ -> СКОПКУ
    new_s = []
    for c1, c2 in zip(xword, xword[1:]):
        if c2 in 'кпстфх':
            new_s.append(accentuator.conson(c1))
        else:
            new_s.append(c1)
    xword = ''.join(new_s) + xword[-1]

    #
    #         # последнюю согласную оглушаем всегда:
    #         # ГОД -> ГОТ
    #         new_s.append(self.conson(s[-1]))
    #
    #         s = ''.join(new_s)


    # огрушаем последнюю согласную с мягким знаком:
    # ВПРЕДЬ -> ВПРЕТЬ
    if len(xword) >= 2 and xword[-1] == 'ь' and xword[-2] in 'бвгдз':
        xword = xword[:-2] + accentuator.conson(xword[-2]) + 'ь'

    if '^' in xword:
        apos = xword.index('^')
        if apos == len(xword) - 2:
            # ударная гласная - последняя, в этом случае включаем предшествующую букву.
            # 07.12.2022 но если гласная идет после "ь" или "ъ", как в "ружья"
            #                                                              ^^
            if xword[apos-1] in 'ьъ':
                clausula = xword[apos-2:]
            else:
                clausula = xword[apos-1:]
        else:
            clausula = xword[apos:]
    else:
        clausula = xword

    return xword, clausula


def rhymed_fuzzy(accentuator, word1, stress1, ud_tags1, word2, stress2, ud_tags2):
    return rhymed_fuzzy2(accentuator, word1, stress1, ud_tags1, '', None, word2, stress2, ud_tags2, '', None)


def rhymed_fuzzy2(accentuator, word1, stress1, ud_tags1, unstressed_prefix1, unstressed_tail1, word2, stress2, ud_tags2, unstressed_prefix2, unstressed_tail2):
    if stress1 is None:
        stress1 = accentuator.get_accent(word1, ud_tags1)

    if stress2 is None:
        stress2 = accentuator.get_accent(word2, ud_tags2)

    word1 = accentuator.yoficate2(accentuator.sanitize_word(word1), ud_tags1)
    word2 = accentuator.yoficate2(accentuator.sanitize_word(word2), ud_tags2)

    k = (word1, stress1, unstressed_prefix1, unstressed_tail1, word2, stress2, unstressed_prefix2, unstressed_tail2,)
    res1 = accentuator.fuzzy_rhyming_cache.get(k, None)
    if res1 is not None:
        return res1

    xword1, clausula1 = render_xword(accentuator, word1, stress1, ud_tags1, unstressed_prefix1, unstressed_tail1)
    xword2, clausula2 = render_xword(accentuator, word2, stress2, ud_tags2, unstressed_prefix2, unstressed_tail2)
    #res1 = accentuator.fuzzy_rhyming_cache.get((xword1, xword2), None)
    #if res1 is not None:
    #    return res1

    res = rhymed_fuzzy2_base(accentuator, word1, stress1, xword1, clausula1, word2, stress2, xword2, clausula2)
    #accentuator.fuzzy_rhyming_cache[(xword1, xword2)] = res
    accentuator.fuzzy_rhyming_cache[k] = res
    return res


def rhymed_fuzzy2_base(accentuator, word1, stress1, xword1, clausula1, word2, stress2, xword2, clausula2):
    #print(f'\nDEBUG@1999 word1={word1} xword1={xword1} clausula1={clausula1}  word2={word2} xword2={xword2} clausula2={clausula2}\n')
    #print(f'\nDEBUG@2000 xword1={xword1} xword2={xword2}  word1={word1} word2={word2}\n')

    if len(clausula1) >= 3 and clausula1 == clausula2:
        # клаузуллы достаточно длинные и совпадают:
        # поэтом - ответом
        return True

    phonetic_consonants = 'бвгджзклмнпрстфхцчшщ'
    phonetic_vowels = 'аеёиоуыэюя'

    e_2_je = {'э': 'е', 'у': 'ю', 'а': 'я', 'о': 'ё'}

    # TODO: нижеследующий код переделать на левенштейна с кастомными весами операций!
    #
    for c1, c2 in [(clausula1, clausula2), (clausula2, clausula1)]:
        # 05.12.2022 Ситуация, когда clausula1 и clausula2 имеют одинаковое начало, но одна из них длиннее на 1 согласную:
        # ВЕРИЛ - ДВЕРИ
        if len(c1) == len(c2)+1 and c1.startswith(c2) and c1[-1] in phonetic_consonants:
            return True

        # clausula1 и clausula2 имеют одинаковое начало, но оканчиваются на разную гласную:
        # ВЛОЖЕНЫ - ПОЛОЖЕНО
        if len(c1) == len(c2) and c1[:-1] == c2[:-1] and len(c1) >= 3 \
                and c1[-2] in phonetic_consonants and c1[-1] in phonetic_vowels and c2[-1] in phonetic_vowels:
            return True

        # С^ИНИЙ - ОС^ИНА
        #  ^^^^^     ^^^^
        if len(c1) == 5 and len(c2) == 4 and c1.startswith(c2[:3]) \
            and c1[-2] in phonetic_vowels and c1[-1] == 'й' \
            and c2[-1] in phonetic_vowels:
            return True

        # АРЛЕК^ИНО - ЖУРАВЛ^ИНЫХ
        #      ^^^^         ^^^^^
        if len(c1) == 4 and len(c2) == 5 and c1[:3] == c2[:3] \
            and c1[-1] in phonetic_vowels \
            and c2[-2] in phonetic_vowels and c2[-1] in phonetic_consonants:
            return True

        # УС^АТЫЙ - ВОЛЧ^АТАМ
        #   ^^^^^       ^^^^^
        if len(c1) == len(c2) and len(c1) >= 4 and c1[:-2] == c2[:-2] \
            and c1[-2] in phonetic_vowels and c2[-2] in phonetic_vowels \
            and c1[-1] in phonetic_consonants+'й' and c2[-1] in phonetic_consonants+'й':
            return True

        # ЖЕМЧ^УЖНОЮ - ^ЮЖНУЮ
        if len(c1) == len(c2) and len(c1) >= 4 and c1[:-2] == c2[:-2] and c1[-1] == c2[-1] \
            and c1[-2] in phonetic_vowels and c2[-2] in phonetic_vowels:
            return True

        # ^Я - РУЖЬ^Я
        if c1 == '^я' and len(c2) == 4 and c2.endswith('ь^я'):
            return True

        # МО^Я - РУЖЬ^Я
        if len(c1)>=3 and len(c2)>=3 and c1[-2:] == c2[-2:] and c1[-2] == '^' and c1[-1] in 'еёяю' and c2[-1] in 'еёюя' \
            and c1[-3] in phonetic_vowels+'ьъ' and c2[-3] in phonetic_vowels+'ьъ':
            return True

        # УПО^ЕНЬЯ - НАСТРО^ЕНЬЕ
        #    ^^^^^         ^^^^^
        if len(c1) == len(c2) and len(c1) >= 5 and c1[:-1] == c2[:-1] \
            and c1[-2] in 'ьъ' and c1[-1] in phonetic_vowels and c2[-1] in phonetic_vowels:
            return True

        # Н^ЕТ - ЛЮДО^ЕД
        #  ^^^       ^^^
        if len(c1) == 3 and len(c2) == 3 and c1[0] == '^' and c2[0] == '^' and c1[1] == 'э' and c2[1] == 'е' and c1[2] == c2[2]:
            return True

        # ПОБ^ЕДА - ПРИ^ЕДУ
        #    ^^^^      ^^^^
        if len(c1) == 4 and len(c2) == 4 and c1[0] == '^' and c1[1] in 'аоуэ' and c2[1] in e_2_je.get(c1[1], []) \
            and c1[2] in phonetic_consonants and c1[2] == c2[2] \
            and c1[-1] in phonetic_vowels and c2[-1] in phonetic_vowels:
            return True

    for r1, r2, rx_text1, rx_text2, num_groups in fuzzy_ending_pairs:
         if num_groups > 0:
             if check_ending_rx_matching_3(xword1, xword2, r1, r2, rx_text1, rx_text2, num_groups):
                 return True
         else:
            if check_ending_rx_matching_2(xword1, xword2, r1, r2):
                #print('\nDEBUG@859 word1={} rx={}  <==>  word2={} rx={}\n'.format(xword1, s1, xword2, s2))
                return True

            if check_ending_rx_matching_2(xword1, xword2, r2, r1):
                #print('\nDEBUG@863 word1={} rx={}  <==>  word2={} rx={}\n'.format(xword1, s2, xword2, s1))
                return True

    if accentuator.allow_rifmovnik and len(word1) >= 2 and len(word2) >= 2:
        eword1, keys1 = extract_ekeys(word1, stress1)
        eword2, keys2 = extract_ekeys(word2, stress2)
        for key1 in keys1:
            if key1 in accentuator.rhyming_dict:
                for key2 in keys2:
                    if key2 in accentuator.rhyming_dict[key1]:
                        #print('\nDEBUG@1006 for word word1="{}" word2="{}"\n'.format(word1, word2))
                        return True

    return False



def extract_ekeys(word, stress):
    cx = []
    vcount = 0
    stressed_c = None
    for c in word:
        if c in 'аеёиоуыэюя':
            vcount += 1
            if vcount == stress:
                stressed_c = c.upper()
                cx.append(stressed_c)
            else:
                cx.append(c)
        else:
            cx.append(c)

    word1 = ''.join(cx)
    keys1 = []
    eword1 = None
    for elen in range(2, len(word1)):
        eword1 = word1[-elen:]
        if eword1[0] == stressed_c or eword1[1] == stressed_c:
            keys1.append(eword1)
    return eword1, keys1


if __name__ == '__main__':
    proj_dir = os.path.expanduser('~/polygon/text_generator')
    data_folder = os.path.join(proj_dir, 'data/poetry/dict')
    tmp_dir = os.path.join(proj_dir, 'tmp')

    accents = Accents(device="cpu")
    accents.load(data_folder, None)
    accents.save_pickle(os.path.join(proj_dir, 'models', 'accentuator', 'accents.pkl'))


