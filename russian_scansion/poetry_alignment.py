"""
This module provides tools for analyzing Russian poetry, including:
- Stress mark placement
- Meter detection
- Rhyme scheme analysis
- Technical quality assessment

It is part of the **Poetry Scansion Tool** project.
Repository: https://github.com/Koziev/RussianPoetryScansionTool

15-12-2021 Введен сильный штраф за 2 подряд ударных слога
17-12-2021 Регулировка ударности некоторых наречий и частиц, удаление лишних пробелов вокруг дефиса при выводе строки с ударениями
18-12-2021 Коррекция пробелов вынесена в отдельный модуль whitespace_normalization
22-12-2021 Добавлена рифмовка AABB
28-12-2021 Добавлены еще штрафы за всякие нехорошие с точки зрения вокализма ситуации в строке, типа 6 согласных подряд в смежных словах "пёстр страх"
23-01-2022 Добавлен код для словосочетаний с вариативным ударением типа "пО лесу | по лЕсу"
26-01-2022 Если слово допускает альтернативные ударения по списку и теги не позволяют сделать выбор, то берем первое ударение, а не бросаем исключение.
07-04-2022 Если из-за ошибки частеречной разметки не удалось определить вариант ударения омографа, то будем перебирать все варианты.
22-04-2022 отдельно детектируем рифмовку AAAA, так как она зачастую выглядит очень неудачно и ее желательно устранять из обучающего датасета.
07.06.2022 не штрафуем строку с одним ударным слогом, если строка состоит из единственного слова или сочетания предлог+сущ
10.06.2022 Если в строке есть только одно слово (без учета пунктуации), то для него берем все известные варианты ударения. Это нужно
           для корректной разметки депрессяшек/артишоков, так как частеречная разметка на одном слове не работает и не позволяет
           выбрать корректный вариант ударения.
22.06.2022 в артишоках для последней строки с одним словом для OOV делаем перебор всех вариантов ударности.
04.08.2022 добавлен учет 3-словных словосочетаний типа "бок О бок"
06.12.2022 полная переработка алгоритма расстановк ударений: оптимизация, подготовка к использованию спеллчекера
09.12.2022 Тесты ударятора вынесены в отдельный файл.
10.10.2023 Добавляем выравнивание 5, 6 и 7-строчников
06.10.2024 Добавляется работа со вторичными ударениями
07.10.2024 Все данные загружаются из pickle-файла
"""

import collections
import itertools
import pathlib
import traceback
from functools import reduce
import os
import random
import io
import math
import jellyfish
import re
import numpy as np
import pickle
from typing import List, Set, Dict, Tuple, Optional

from .phonetic import Accents, rhymed2, rhymed_fuzzy2, render_xword, WordAccentuation
from .metre_classifier import get_syllables
from .whitespace_normalization import normalize_whitespaces


# Коэффициенты для штрафов за разные отступления от идеальной метрики.
COEFF = dict()
COEFF['@68'] = 0.95  # 0.5
COEFF['@68_2'] = 0.98  # 0.95
COEFF['@71'] = 1.0
COEFF['@75'] = 0.98  # 0.9
COEFF['@77'] = 1.0
COEFF['@77_2'] = 1.0
COEFF['@79'] = 1.0
COEFF['@126'] = 0.98
COEFF['@225'] = 0.95
COEFF['@143'] = 0.9


def tokenize(s):
    return [token for token in re.split(r'[.,!?\- ;:…\n"«»]', s) if token]


def mul(items):
    return reduce((lambda x, y: x * y), items)


class Defect(object):
    def __init__(self, penalty, description):
        self.penalty = penalty
        self.description = description

    def __repr__(self):
        return self.description + '({})'.format(self.penalty)

    def serialize(self):
        return {'penalty': self.penalty, 'description': self.description}


class Defects(object):
    def __init__(self):
        self.items = []

    def __repr__(self):
        s = '({:5.2f})'.format(self.get_cumulative_factor())
        if self.items:
            s += ' {}'.format('; '.join(map(str, self.items)))

        return s

    def add_defect(self, defect):
        self.items.append(defect)

    def has_defects(self):
        return len(self.items) > 0

    def get_cumulative_factor(self):
        if self.items:
            return mul((1.0-defect.penalty) for defect in self.items)
        else:
            return 1.0

    def serialize(self):
        return {'cumulative_factor': self.get_cumulative_factor(), 'items': [d.serialize() for d in self.items]}


class MetreMappingResult(object):
    def __init__(self, prefix, metre_signature):
        self.score = 1.0
        self.word_mappings = []
        self.stress_shift_count = 0
        self.prefix = prefix
        self.metre_signature = metre_signature
        self.cursor = 0

    def enforce_intrinsic_word_accentuation(self):
        self.word_mappings = [WordMappingResult.build_from_word_stress_variant(word_mapping.word) for word_mapping in self.word_mappings]

    def count_prev_unstressed_syllables(self):
        num_unstressed_syllables = 0
        for word_mapping in self.word_mappings[::-1]:
            if word_mapping.word.new_stress_pos == -1:
                num_unstressed_syllables += word_mapping.word.poetry_word.n_vowels
            else:
                break
        return num_unstressed_syllables

    @staticmethod
    def build_from_nonpoetry(line):
        r = MetreMappingResult(prefix=0, metre_signature=None)
        for word in line.stressed_words:
            word_mapping = WordMappingResult.build_from_word_stress_variant(word)
            r.word_mappings.append(word_mapping)
        return r


    @staticmethod
    def build_for_empty_line():
        r = MetreMappingResult(prefix=0, metre_signature=None)
        return r

    @staticmethod
    def build_from_source(src_mapping, new_cursor):
        new_mapping = MetreMappingResult(src_mapping.prefix, src_mapping.metre_signature)
        new_mapping.score = src_mapping.score
        new_mapping.word_mappings = list(src_mapping.word_mappings)
        new_mapping.stress_shift_count = src_mapping.stress_shift_count
        new_mapping.cursor = new_cursor
        return new_mapping

    def add_word_mapping(self, word_mapping):
        self.word_mappings.append(word_mapping)
        self.score *= word_mapping.get_total_score()
        if word_mapping.stress_shift:
            self.stress_shift_count += 1

    def finalize(self):
        # ищем цепочки безударных слогов (000...) длиннее 3х подряд, и штрафуем.
        signature = list(itertools.chain(*[m.word.stress_signature for m in self.word_mappings]))
        s = ''.join(map(str, signature))
        for m in re.findall(r'(0{4,})', s):
            factor = 0.1

            # безударный промежуток внутри строки - то есть слева и справа есть ударные слоги
            l = len(m)
            i = s.index(m)
            if l <= 5 and '1' in s[:i] and '1' in s[i+l:]:
                swi = list(itertools.chain(*[[i]*len(m.word.stress_signature) for i, m in enumerate(self.word_mappings)]))
                if len(set(swi[i: i+l])) <= 2:
                    factor = {4: 0.80, 5: 0.50}[l]
                else:
                    factor = {4: 0.30, 5: 0.20}[l]

            self.score *= factor

        return

    def count_stress_marks(self) -> int:
        n = sum(word_mapping.count_stress_marks() for word_mapping in self.word_mappings)
        return n

    def get_stressed_line(self) -> str:
        s = ' '.join(word_mapping.render_accentuation() for word_mapping in self.word_mappings)
        s = normalize_whitespaces(s)
        return s

    def get_stress_signature_str(self) -> str:
        return ''.join(word_mapping.get_stress_signature_str() for word_mapping in self.word_mappings)

    def __repr__(self):
        if self.word_mappings:
            sx = []

            for word_mapping in self.word_mappings:
                sx.append(str(word_mapping))

            sx.append('〚' + '{:6.2g}'.format(self.score).strip() + '〛')
            return ' '.join(sx)
        else:
            return '««« EMPTY »»»'

    def get_score(self):
        stress_shift_factor = 1.0 if self.stress_shift_count < 2 else pow(0.5, self.stress_shift_count)
        return self.score * stress_shift_factor # * self.src_line_variant_score

    def get_canonic_meter(self):
        if self.metre_signature is None:
            return ''
        elif self.metre_signature == (0, 1):
            return 'ямб' if self.prefix == 0 else 'хорей'
        elif self.metre_signature == (1, 0):
            return 'хорей' if self.prefix == 0 else 'ямб'
        elif len(self.metre_signature) == 3:
            m = list(self.metre_signature)
            if self.prefix == 1:
                m = m[-1:] + m[:-1]
            m = tuple(m)
            if m == (1, 0, 0):
                return 'дактиль'
            elif m == (0, 1, 0):
                return 'амфибрахий'
            elif m == (0, 0, 1):
                return 'анапест'
            else:
                raise NotImplementedError()
        else:
            return ''


class WordMappingResult(object):
    def __init__(self, word, TP, FP, TN, FN, syllabic_mapping, stress_shift, additional_score_factor):
        self.word = word
        self.TP = TP
        self.FP = FP
        self.TN = TN
        self.FN = FN
        self.syllabic_mapping = syllabic_mapping
        self.metre_score = pow(0.1, FP) * pow(0.95, FN) * additional_score_factor
        self.total_score = self.metre_score * word.get_score()
        self.stress_shift = stress_shift

    @staticmethod
    def build_from_word_stress_variant(word):
        syllabic_mapping = []
        TP = 0  # кол-во ударных слогов
        TN = 0  # кол-во безударных слогов
        for s in word.stress_signature:
            if s in (1, 2):
                syllabic_mapping.append('TP')
                TP += 1
            elif s == 0:
                syllabic_mapping.append('TN')
                TN += 1

        r = WordMappingResult(word=word,
                              TP=TP, FP=0, TN=TN, FN=0,
                              syllabic_mapping=syllabic_mapping,
                              stress_shift=0,
                              additional_score_factor=1.0)
        return r

    def get_total_score(self):
        return self.total_score

    def count_stress_marks(self) -> int:
        return self.syllabic_mapping.count('TP')

    def get_stress_signature_str(self) -> str:
        rendering = []

        syllable_index = 0
        for c in self.word.form:
            if c.lower() in 'аеёиоуыэюя':
                if self.syllabic_mapping[syllable_index] == 'TP':
                    # Тут ударение, основное или вторичное.
                    rendering.append(self.word.stress_signature[syllable_index])
                elif self.syllabic_mapping[syllable_index] == 'FP':
                    # Тут в слове есть ударение, но в метре тут безударный слог.
                    #rendering.append(self.word.stress_signature[syllable_index])
                    rendering.append(0)
                else:
                    # Ударения нет.
                    rendering.append(0)

                syllable_index += 1

        return ''.join(map(str, rendering))

    def render_accentuation(self) -> str:
        rendering = []

        syllable_index = 0
        for c in self.word.form:
            if c.lower() in 'аеёиоуыэюя':
                rendering.append(c)
                if self.syllabic_mapping[syllable_index] == 'TP':
                    # Тут ударение. Посмотрим, это основное или вторичное.
                    if self.word.stress_signature[syllable_index] == 2:
                        # вторичное
                        rendering.append('\u0300')
                    elif self.word.stress_signature[syllable_index] == 1:
                        # основное
                        rendering.append('\u0301')
                    else:
                        raise NotImplementedError()

                syllable_index += 1
            else:
                rendering.append(c)

        return ''.join(rendering)

    def __repr__(self):
        #s = self.word.get_stressed_form()
        s = self.render_accentuation()
        if self.total_score != 1.0:
            s += '[' + '{:5.2g}'.format(self.total_score).strip() + ']'
        return s


class MetreMappingCursor(object):
    def __init__(self, metre_signature: List[int], prefix: int):
        self.prefix = prefix
        self.metre_signature = metre_signature
        self.length = len(metre_signature)

    def get_stress(self, cursor) -> int:
        """Возвращает ударность, ожидаемую в текущей позиции"""
        if self.prefix:
            if cursor == 0:
                return 0
            else:
                return self.metre_signature[(cursor - self.prefix) % self.length]
        else:
            return self.metre_signature[cursor % self.length]

    def map(self, stressed_words_chain, aligner):
        start_results = [MetreMappingResult(self.prefix, self.metre_signature)]
        final_results = []
        self.map_chain(prev_node=stressed_words_chain, prev_results=start_results, aligner=aligner, final_results=final_results)
        final_results = sorted(final_results, key=lambda z: -z.get_score())
        return final_results

    def map_chain(self, prev_node, prev_results, aligner, final_results):
        for cur_slot in prev_node.next_nodes:
            cur_results = self.map_word(stressed_word_group=cur_slot.stressed_words, results=prev_results, aligner=aligner)
            if cur_slot.next_nodes:
                self.map_chain(prev_node=cur_slot, prev_results=cur_results, aligner=aligner, final_results=final_results)
            else:
                for result in cur_results:
                    result.finalize()
                    final_results.append(result)

    def map_word(self, stressed_word_group, results: [MetreMappingResult], aligner):
        new_results = []

        for prev_result in results:
            for word_mapping, new_cursor in self.map_word1(stressed_word_group, prev_result, aligner):
                if word_mapping.word.new_stress_pos == -1:
                    # Пресекаем появление цепочек из безударных слогов длиной более 4.
                    n = prev_result.count_prev_unstressed_syllables()
                    if word_mapping.word.poetry_word.n_vowels + n >= 6:  # >= 4
                        continue
                next_metre_mapping = MetreMappingResult.build_from_source(prev_result, new_cursor)
                next_metre_mapping.add_word_mapping(word_mapping)
                new_results.append(next_metre_mapping)

        new_results = sorted(new_results, key=lambda z: -z.get_score())

        return new_results

    def map_word1(self, stressed_word_group, result: MetreMappingResult, aligner):
        result_mappings = []

        for stressed_word in stressed_word_group:
            cursor = result.cursor
            TP, FP, TN, FN = 0, 0, 0, 0
            syllabic_mapping = []
            for word_sign in stressed_word.stress_signature:
                metre_sign = self.get_stress(cursor)
                if metre_sign == 1:
                    if word_sign == 1:
                        # Ударение должно быть и оно есть
                        TP += 1
                        syllabic_mapping.append('TP')
                    elif word_sign == 2:
                        # Ударение должно быть, и есть слабое на этом слоге
                        TP += 0.5
                        syllabic_mapping.append('TP')
                    elif word_sign == 0:
                        # ударение должно быть, но его нет
                        FN += 1
                        syllabic_mapping.append('FN')
                    else:
                        raise RuntimeError()
                else:
                    if word_sign == 1:
                        # Ударения не должно быть, но оно есть
                        FP += 1
                        syllabic_mapping.append('FP')
                    elif word_sign == 2:
                        # Ударения не должно быть, и в этом месте оно слабое.
                        TN += 1
                        syllabic_mapping.append('TN')
                    elif word_sign == 0:
                        # Ударения не должно быть, и его нет
                        TN += 1
                        syllabic_mapping.append('TN')
                    else:
                        raise RuntimeError()
                cursor += 1

            # Проверим сочетание ударения в предыдущем слове и в текущем, в частности - оштрафуем за два ударных слога подряд
            additional_score_factor = 1.0
            if len(stressed_word.stress_signature) > 0:
                if len(result.word_mappings) > 0:
                    prev_mapping = result.word_mappings[-1]
                    if prev_mapping.word.stress_signature:
                        if prev_mapping.word.stress_signature[-1] == 1:  # предыдущее слово закончилось ударным слогом
                            if stressed_word.stress_signature[0] == 1:
                                # большой штраф за два ударных подряд
                                additional_score_factor = 0.1

            mapping1 = WordMappingResult(stressed_word,
                                         TP, FP, TN, FN,
                                         syllabic_mapping=syllabic_mapping,
                                         stress_shift=False,
                                         additional_score_factor=additional_score_factor)
            result_mappings.append((mapping1, cursor))

        return result_mappings


class WordStressVariant(object):
    def __init__(self, poetry_word, new_stress_pos, score, secondary_accentuation=None):
        self.poetry_word = poetry_word
        self.new_stress_pos = new_stress_pos
        self.score = score

        self.stress_signature = []
        output = []
        n_vowels = 0
        for c in self.poetry_word.form:
            output.append(c)
            if c.lower() in 'уеыаоэёяию':
                n_vowels += 1
                if n_vowels == self.new_stress_pos:
                    output.append('\u0301')
                    self.stress_signature.append(1)
                else:
                    if secondary_accentuation is not None and secondary_accentuation[n_vowels-1] == 2:
                        self.stress_signature.append(2)
                        output.append('\u0300')
                    else:
                        self.stress_signature.append(0)
        self.stressed_form = ''.join(output)

        self.is_cyrillic = self.poetry_word.form[0].lower() in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'

    def get_score(self):
        return self.score

    def count_syllables(self) -> int:
        return len(self.stress_signature)

    @property
    def form(self):
        return self.poetry_word.form

    def build_stressed(self, new_stress_pos):
        return WordStressVariant(self.poetry_word, new_stress_pos, self.score)

    def build_unstressed(self):
        return WordStressVariant(self.poetry_word, -1, self.score)

    def get_stressed_form(self, show_secondary_accentuation):
        output = []
        n_vowels = 0
        for c in self.poetry_word.form:
            output.append(c)
            if c.lower() in 'уеыаоэёяию':
                stress = self.stress_signature[n_vowels]
                if stress == 1:
                    output.append('\u0301')
                elif stress == 2 and show_secondary_accentuation:
                    output.append('\u0300')
                n_vowels += 1

        return ''.join(output)

    def is_short_word(self):
        if self.is_cyrillic:
            return len(self.poetry_word.form) <= 2
        else:
            return False

    def __repr__(self):
        s = self.stressed_form
        if self.score != 1.0:
            s += '({:5.3f})'.format(self.score)
        return s

    def split_to_syllables(self):
        output_syllables = []

        sx = get_syllables(self.poetry_word.form)
        if sx:
            vcount = 0
            for syllable in sx:
                syllable2 = []
                for c in syllable.text:
                    syllable2.append(c)
                    if c.lower() in 'уеыаоэёяию':
                        vcount += 1
                        if vcount == self.new_stress_pos:
                            syllable2.append('\u0301')
                syllable2 = ''.join(syllable2)
                output_syllables.append(syllable2)
        else:
            vcount = 0
            syllable2 = []
            for c in self.poetry_word.form:
                syllable2.append(c)
                if c.lower() in 'уеыаоэёяию':
                    vcount += 1
                    if vcount == self.new_stress_pos:
                        syllable2.append('\u0301')
            syllable2 = ''.join(syllable2)
            output_syllables.append(syllable2)

        return output_syllables


class PoetryWord(object):
    def __init__(self, lemma, form, upos, tags, accentuations):
        self.lemma = lemma
        self.form = form
        self.upos = upos
        self.tags = tags
        self.tags2 = dict(s.split('=') for s in tags)
        self.accentuations = list(accentuations)
        self.stress_pos = accentuations[0].stress_pos if accentuations else -1

        self.is_rhyming_word = False  # отмечаем последнее слово в каждой строке

        self.leading_consonants = 0  # кол-во согласных ДО первой гласной
        self.trailing_consonants = 0  # кол-во согласных ПОСЛЕ последней гласной
        n_vowels = 0
        for c in self.form:
            clower = c.lower()
            if clower in 'уеыаоэёяию':
                self.trailing_consonants = 0
                n_vowels += 1
            elif clower in 'бвгджзклмнпрстфхцчшщ':
                if n_vowels == 0:
                    self.leading_consonants += 1
                else:
                    self.trailing_consonants += 1
        self.n_vowels = n_vowels

    def get_attr(self, tag_name):
        return self.tags2.get(tag_name, '')

    def __repr__(self):
        if self.accentuations:
            # Для визуализации берем первый вариант акцентуации
            accentuation1 = self.accentuations[0]

            output = []
            n_vowels = 0
            for c in self.form:
                output.append(c)
                if c.lower() in 'уеыаоэёяию':
                    n_vowels += 1
                    if n_vowels == accentuation1.stress_pos:
                        output.append('\u0301')
                    elif accentuation1.secondary_accentuation and n_vowels in accentuation1.secondary_accentuation:
                        output.append('\u0300')
            return ''.join(output)
        else:
            return self.form

    def get_stress_variants(self, aligner, allow_stress_shift, allow_unstress12):
        variants = []

        nvowels = self.n_vowels  # count_vowels(self.form.lower())
        uform = self.form.lower()

        if aligner.allow_stress_guessing_for_oov and aligner.accentuator.is_oov(uform) and nvowels > 1:
            # 17.08.2022 для OOV слов просто перебираем все варианты ударения.
            # в будущем тут можно вызвать модельку в ударяторе, которая выдаст вероятности ударения на каждой из гласных.
            vowel_count = 0
            for i, c in enumerate(uform):
                if c in 'уеыаоэёяию':
                    vowel_count += 1

                    proba = 0.90  #1. / nvowels

                    # для слов с 4 и более гласными предпочитаем такой вариант ударения, при котором
                    # не возникает подряд 3 безударных слога слева или справа от текущей гласной.
                    # Например: "коготочков"
                    if count_vowels(uform[:i]) > 2 or count_vowels(uform[i+1:]) > 2:
                        proba *= 0.5

                    variants.append(WordStressVariant(self, vowel_count, proba))
        elif uform == 'начала' and self.upos == 'NOUN':
            # У этого слова аж 3 варианта ударения, два для глагола и 1 для существительного.
            # Если тэггер считает, что у нас именно существительное - не используем вариативность глагола.
            # TODO вынести такую логику как-то в словарь ambiguous_accents_2.yaml, чтобы не приходилось тут
            # хардкодить.
            variants.append(WordStressVariant(self, 2, 1.0))
        #elif uform in aligner.accentuator.ambiguous_accents2:
        #    # Вариативное ударение в рамках одной грамматической формы пОнял-понЯл
        #    for stress_pos in aligner.accentuator.ambiguous_accents2[uform]:
        #        variants.append(WordStressVariant(self, stress_pos, 1.0))
        elif len(self.accentuations) > 1:
            # 07.04.2022 из-за ошибки pos-tagger'а не удалось обработать омограф.
            # будем перебирать все варианты ударения в нем.
            for accentuation in self.accentuations:
                variants.append(WordStressVariant(self,
                                                  new_stress_pos=accentuation.stress_pos,
                                                  secondary_accentuation=accentuation.secondary_accentuation,
                                                  score=1.0))
        elif uform == 'нибудь':
            # Частицу "нибудь" не будем ударять:
            # Мо́жет что́ - нибу́дь напла́чу
            #             ~~~~~~
            variants.append(WordStressVariant(self, -1, 1.0))
            if self.is_rhyming_word:
                # уколи сестра мне
                # в глаз чего нибудь     <======
                # за бревном не вижу
                # к коммунизму путь
                variants.append(WordStressVariant(self, self.stress_pos, 1.0))
            else:
                variants.append(WordStressVariant(self, self.stress_pos, 0.5))
        elif nvowels == 1 and self.upos in ('NOUN', 'NUM', 'ADJ') and allow_unstress12:
            # Односложные слова типа "год" или "два" могут стать безударными:
            # В год Петуха́ учи́тесь кукаре́кать.
            #   ^^^
            # Оле́г два дня́ прожи́л без во́дки.
            #      ^^^
            variants.append(WordStressVariant(self, self.stress_pos, 1.0))
            variants.append(WordStressVariant(self, -1, 0.7))
        elif nvowels == 1 and self.upos == 'VERB' and allow_unstress12:
            # 21.08.2022 разрешаем становится безударными односложным глаголам.
            variants.append(WordStressVariant(self, self.stress_pos, 1.0))
            variants.append(WordStressVariant(self, -1, 0.7))
        elif uform == 'нет':  # and not self.is_rhyming_word:
            # частицу (или глагол для Stanza) "нет" с ударением штрафуем
            variants.append(WordStressVariant(self, self.stress_pos, COEFF['@68_2']))

            # а вариант без ударения - с нормальным скором:
            variants.append(WordStressVariant(self, -1, COEFF['@71']))
        elif self.upos in ('ADP', 'CCONJ', 'SCONJ', 'PART', 'INTJ', 'AUX') and uform not in ('словно', 'разве'): # and not self.is_rhyming_word:
            if uform in ('о', 'у', 'из', 'от', 'под', 'подо', 'за', 'при', 'до', 'про', 'для', 'ко', 'со', 'во', 'на', 'по', 'об', 'обо', 'без', 'над', 'пред') and self.upos == 'ADP':
                # эти предлоги никогда не делаем ударными
                variants.append(WordStressVariant(self, -1, 1.0))

                # Но если это последнее слово в строке, то допускается вариант:
                # необосно́ванная ве́ра
                # в своё́ владе́ние дзюдо́
                # не так вредна́ в проце́ссе дра́ки
                # как до
                if self.is_rhyming_word:
                    variants.append(WordStressVariant(self, self.stress_pos, 1.0))
                else:
                    # Снова скучная осень заводит свои хороводы
                    # Из опавшей листвы и занудливых серых дождей.
                    # Ты бредешь средь толпы, так похожая на непогоду,   <===
                    # Под широким зонтом по асфальту сырых площадей.
                    variants.append(WordStressVariant(self, self.stress_pos, 0.5))
            elif uform in ('не',):
                variants.append(WordStressVariant(self, -1, 1.0))
                if self.is_rhyming_word:
                    variants.append(WordStressVariant(self, self.stress_pos, 0.95))
                else:
                    variants.append(WordStressVariant(self, self.stress_pos, 0.20))
            elif uform in ('бы', 'ли', 'же', 'ни', 'ка'):
                # Частицы "бы" и др. никогда не делаем ударной
                variants.append(WordStressVariant(self, -1, 1.0))

                if self.is_rhyming_word:
                    variants.append(WordStressVariant(self, self.stress_pos, COEFF['@68']))
            elif uform in ('а', 'и', 'или', 'но'):
                # союзы "а", "и", "но" обычно безударный:
                # А была бы ты здорова
                # ^
                variants.append(WordStressVariant(self, -1, 1.0))

                if self.is_rhyming_word:
                    # ударный вариант в рифмуемой позиции
                    variants.append(WordStressVariant(self, self.stress_pos, 0.70))  # ударный вариант
                else:
                    variants.append(WordStressVariant(self, self.stress_pos, 0.20))  # ударный вариант
            elif uform == 'и' and self.upos == 'PART':
                # Частицу "и" не делаем ударной:
                # Вот и она... Оставив магазин
                #     ^
                variants.append(WordStressVariant(self, -1, 1.0))

                if self.is_rhyming_word:
                    variants.append(WordStressVariant(self, self.stress_pos, COEFF['@68']))
            elif uform in ('были', 'было'):
                variants.append(WordStressVariant(self, -1, 1.0))
                variants.append(WordStressVariant(self, self.stress_pos, 1.0))
            elif nvowels == 0:
                # Предлоги без единой гласной типа "в"
                variants.append(WordStressVariant(self, -1, 1.0))
            elif uform in ['передо']:
                # Безударный вариант:
                #
                # Из вечной ночи призываю вновь
                # Друзей, давно ушедших безвозвратно,
                # И предстает передо мной любовь,
                # Которую уж не вернуть обратно.

                variants.append(WordStressVariant(self, -1, 0.7))
            else:
                if count_vowels(uform) < 3:
                    # Предлоги, союзы, частицы предпочитаем без ударения, если в них меньше трех гласных.

                    # Сначала добавляем вариант без ударения - с нормальным скором:
                    variants.append(WordStressVariant(self, -1, COEFF['@71']))

                    # второй вариант с ударением добавляем с дисконтом:
                    if uform in ['лишь', 'вроде', 'если', 'чтобы', 'когда', 'просто', 'мимо', 'даже', 'всё', 'хотя', 'едва', 'нет', 'пока']:
                        variants.append(WordStressVariant(self, self.stress_pos, COEFF['@68_2']))
                    elif uform in ('был', 'будь', 'будем'):
                        variants.append(WordStressVariant(self, self.stress_pos, 1.0))
                    else:
                        #variants.append(WordStressVariant(self, self.stress_pos, COEFF['@68']))
                        variants.append(WordStressVariant(self, self.stress_pos, 1.0))
                else:
                    # Частицы типа "НЕУЖЕЛИ" с 3 и более гласными
                    variants.append(WordStressVariant(self, self.stress_pos, 1.0))
        elif self.upos in ('PRON', 'ADV', 'DET') and uform not in ('что',):
            # Для односложных местоимений (Я), наречий (ТУТ, ГДЕ) и слов типа МОЙ, ВСЯ, если они не последние в строке,
            # добавляем вариант без ударения с дисконтом.
            if nvowels == 1:
                variants.append(WordStressVariant(self, self.stress_pos, 1.0))

                # вариант без ударения
                variants.append(WordStressVariant(self, -1, 1.0))
            else:
                # первым идет обычный вариант с ударением.
                variants.append(WordStressVariant(self, self.stress_pos, COEFF['@79']))

                if uform in ['эти', 'эта', 'этот', 'эту', 'это', 'мои', 'твои', 'моих', 'твоих', 'моим', 'твоим', 'моей', 'твоей',
                             'мою', 'твою', 'его', 'ему', 'нему', 'ее', 'её', 'себе', 'меня', 'тебя', 'свою', 'свои', 'своим', 'они', 'она',
                             'уже', 'этом', 'тебе']:
                    # Безударный вариант для таких двусложных прилагательных
                    variants.append(WordStressVariant(self, -1, COEFF['@77_2']))
                elif nvowels == 2 and allow_unstress12:
                    # двусложные наречия - разрешаем без ударения.
                    # Например, первое слово "Снова":
                    #
                    # Снова скучная осень заводит свои хороводы
                    # Из опавшей листвы и занудливых серых дождей.
                    # Ты бредешь средь толпы, так похожая на непогоду,
                    # Под широким зонтом по асфальту сырых площадей.
                    variants.append(WordStressVariant(self, -1, 0.8))
        else:
            # Добавляем первым исходный вариант с ударением
            for accentuation in self.accentuations:
                variants.append(WordStressVariant(self, new_stress_pos=accentuation.stress_pos, secondary_accentuation=accentuation.secondary_accentuation, score=1.0))

            # Второй вариант - безударный
            if uform in ['есть', 'раз', 'быть', 'будь', 'был']:  # and not self.is_rhyming_word:
                # безударный вариант
                variants.append(WordStressVariant(self, -1, COEFF['@143']))
            elif nvowels in (1, 2) and allow_unstress12:
                # 03.05.2024 безударный вариант для остальных двусложных
                variants.append(WordStressVariant(self, -1, 0.8))

        if allow_stress_shift:
            # Сдвигаем ударение вопреки решению на основе частеречной разметки
            uform = self.form.lower()

            if count_vowels(uform) > 1:
                has_different_stresses = uform in aligner.accentuator.ambiguous_accents and uform not in aligner.accentuator.ambiguous_accents2
                if has_different_stresses:
                    # Можем попробовать взять другой вариант ударности слова, считая,
                    # что имеем дело с ошибкой частеречной разметки.
                    sx = list(aligner.accentuator.ambiguous_accents[uform].keys())

                    for stress_form in sx:
                        stress_pos = -1
                        n_vowels = 0
                        for c in stress_form:
                            if c.lower() in 'уеыаоэёяию':
                                n_vowels += 1

                            if c in 'АЕЁИОУЫЭЮЯ':
                                stress_pos = n_vowels
                                break

                        if not any((variant.new_stress_pos == stress_pos) for variant in variants):
                            # Нашли новый вариант ударности этого слова.
                            # Попробуем использовать его вместо выбранного с помощью частеречной разметки.
                            new_stressed_word = WordStressVariant(self, stress_pos, score=0.99)
                            variants.append(new_stressed_word)

        return variants

    def get_first_stress_variant(self, aligner):
        variants = self.get_stress_variants(aligner, allow_stress_shift=False, allow_unstress12=False)
        return variants[0]
        #return WordStressVariant(self, self.stress_pos, score=1.0)


def sum1(arr):
    return sum((x == 1) for x in arr)


class RhymingTail(object):
    def __init__(self, unstressed_prefix, stressed_word, unstressed_postfix_words):
        self.stressed_word = stressed_word
        self.unstressed_postfix_words = unstressed_postfix_words
        self.unstressed_tail = ''.join(w.poetry_word.form for w in unstressed_postfix_words)
        self.prefix = '' if unstressed_prefix is None else unstressed_prefix
        self.ok = stressed_word is not None and stressed_word.new_stress_pos != -1

    def is_ok(self):
        return self.ok

    def is_simple(self):
        return len(self.unstressed_postfix_words) == 0 and len(self.prefix) == 0

    def __repr__(self):
        return self.get_text()

    def get_text(self):
        sx = []
        if self.prefix:
            sx.append(self.prefix)

        if self.stressed_word:
            sx.append(self.stressed_word.stressed_form)

        if self.unstressed_tail:
            sx.extend([' ', self.unstressed_tail])

        return ''.join(sx)

    def get_unstressed_tail(self):
        return self.unstressed_tail


class StressVariantsSlot(object):
    def __init__(self):
        self.stressed_words = None
        self.next_nodes = None

    def __repr__(self):
        s = ''

        if self.stressed_words:
            s += '[ ' + ' | '.join(map(str, self.stressed_words)) + ' ]'
        else:
            s += '∅'

        if self.next_nodes:
            if len(self.next_nodes) == 1:
                s += ' ↦ '
                s += str(self.next_nodes[0])
            else:
                s += ' ⇉ ⦃'
                for i, n in enumerate(self.next_nodes, start=1):
                    s += ' 〚{}〛 {}'.format(i, str(n))
                s += '⦄'

        return s

    def count_variants(self):
        n = len(self.stressed_words) if self.stressed_words else 0
        if self.next_nodes:
            for nn in self.next_nodes:
                n += nn.count_variants()
        return n

    @staticmethod
    def build_next(poetry_words, aligner, allow_stress_shift, allow_unstress12):
        next_nodes = []

        pword = poetry_words[0]

        # Проверяем особые случаи трехсловных словосочетаний, в которых ударение ВСЕГДА падает особым образом:
        # бок О бок
        if len(poetry_words) > 2:
            lword1 = pword.form.lower()
            lword2 = poetry_words[1].form.lower()
            lword3 = poetry_words[2].form.lower()
            key = (lword1, lword2, lword3)
            collocs = aligner.collocations.get(key, None)
            if collocs is not None:
                for colloc in collocs:
                    if colloc.stressed_word_index == 0:
                        # первое слово становится ударным, второе и третье - безударные
                        stressed_word1 = WordStressVariant(poetry_words[0], new_stress_pos=colloc.stress_pos, score=1.0)
                        stressed_word2 = WordStressVariant(poetry_words[1], new_stress_pos=-1, score=1.0)
                        stressed_word3 = WordStressVariant(poetry_words[2], new_stress_pos=-1, score=1.0)
                    elif colloc.stressed_word_index == 1:
                        # первое слово становится безударным, второе - ударное, третье - безударное
                        stressed_word1 = WordStressVariant(poetry_words[0], new_stress_pos=-1, score=1.0)
                        stressed_word2 = WordStressVariant(poetry_words[1], new_stress_pos=colloc.stress_pos, score=1.0)
                        stressed_word3 = WordStressVariant(poetry_words[2], new_stress_pos=-1, score=1.0)
                    else:
                        # первое и второе слово безударные, третье - ударное
                        stressed_word1 = WordStressVariant(poetry_words[0], new_stress_pos=-1, score=1.0)
                        stressed_word2 = WordStressVariant(poetry_words[1], new_stress_pos=-1, score=1.0)
                        stressed_word3 = WordStressVariant(poetry_words[2], new_stress_pos=colloc.stress_pos, score=1.0)

                    next_node = StressVariantsSlot()
                    next_node.stressed_words = [stressed_word1]

                    next_node2 = StressVariantsSlot()
                    next_node2.stressed_words = [stressed_word2]
                    next_node.next_nodes = [next_node2]

                    next_node3 = StressVariantsSlot()
                    next_node3.stressed_words = [stressed_word3]
                    next_node2.next_nodes = [next_node3]

                    if len(poetry_words) > 3:
                        next_node3.next_nodes = StressVariantsSlot.build_next(poetry_words[3:],
                                                                              aligner,
                                                                              allow_stress_shift=allow_stress_shift,
                                                                              allow_unstress12=allow_unstress12)

                    next_nodes.append(next_node)

                return [next_node]

        # Проверяем особый случай двусловных словосочетаний, в которых ударение ВСЕГДА падает не так, как обычно:
        # друг др^уга
        if len(poetry_words) > 1:
            lword1 = pword.form.lower()
            lword2 = poetry_words[1].form.lower()
            key = (lword1, lword2)
            collocs = aligner.collocations.get(key, None)
            if collocs is not None:
                for colloc in collocs:
                    if colloc.stressed_word_index == 0:
                        # первое слово становится ударным, второе - безударное
                        stressed_word1 = WordStressVariant(poetry_words[0], new_stress_pos=colloc.stress_pos, score=1.0)
                        stressed_word2 = WordStressVariant(poetry_words[1], new_stress_pos=-1, score=1.0)
                    else:
                        # первое слово становится безударным, второе - ударное
                        stressed_word1 = WordStressVariant(poetry_words[0], new_stress_pos=-1, score=1.0)
                        stressed_word2 = WordStressVariant(poetry_words[1], new_stress_pos=colloc.stress_pos, score=1.0)

                    next_node = StressVariantsSlot()
                    next_node.stressed_words = [stressed_word1]

                    next_node2 = StressVariantsSlot()
                    next_node2.stressed_words = [stressed_word2]
                    next_node.next_nodes = [next_node2]

                    if len(poetry_words) > 2:
                        next_node2.next_nodes = StressVariantsSlot.build_next(poetry_words[2:],
                                                                              aligner,
                                                                              allow_stress_shift=allow_stress_shift,
                                                                              allow_unstress12=allow_unstress12)

                    next_nodes.append(next_node)

                return next_nodes

        # Самый типичный путь - получаем варианты ударения для слова с их весами.
        next_node = StressVariantsSlot()
        next_node.stressed_words = pword.get_stress_variants(aligner,
                                                             allow_stress_shift=allow_stress_shift,
                                                             allow_unstress12=allow_unstress12)
        if len(poetry_words) > 1:
            next_node.next_nodes = StressVariantsSlot.build_next(poetry_words[1:],
                                                                 aligner,
                                                                 allow_stress_shift=allow_stress_shift,
                                                                 allow_unstress12=allow_unstress12)
        next_nodes.append(next_node)
        return next_nodes

    @staticmethod
    def build(poetry_words, aligner, allow_stress_shift, allow_unstress12):
        start = StressVariantsSlot()
        start.next_nodes = StressVariantsSlot.build_next(poetry_words,
                                                         aligner,
                                                         allow_stress_shift=allow_stress_shift,
                                                         allow_unstress12=allow_unstress12)
        return start


class LineStressVariant(object):
    def __init__(self, poetry_line, stressed_words, aligner):
        self.poetry_line = poetry_line
        self.stressed_words = stressed_words
        self.stress_signature = list(itertools.chain(*(w.stress_signature for w in stressed_words)))
        self.stress_signature_str = ''.join(map(str, self.stress_signature))
        self.init_rhyming_tail()
        if aligner is not None:
            self.score_sequence(aligner)

    @staticmethod
    def build_empty_line():
        return LineStressVariant(PoetryLine(), [], None)

    def is_empty(self) -> bool:
        return self.poetry_line.is_empty()

    def score_sequence(self, aligner):
        self.total_score = reduce(lambda x, y: x*y, [w.score for w in self.stressed_words])
        self.penalties = []

        # 09-08-2022 штрафуем за безударное местоимение после безударного предлога с гласной:
        # я вы́рос в ми́ре где драко́ны
        # съеда́ли на́ших дочере́й
        # а мы расти́ли помидо́ры
        # и ке́тчуп де́лали для них
        #                 ^^^^^^^
        if len(self.stressed_words) >= 2 and \
                self.stressed_words[-1].new_stress_pos == -1 and \
                self.stressed_words[-2].new_stress_pos == -1 and \
                self.stressed_words[-2].poetry_word.upos == 'ADP' and \
                count_vowels(self.stressed_words[-2].poetry_word.form) and \
                self.stressed_words[-1].poetry_word.upos == 'PRON':
            self.total_score *= 0.5
            self.penalties.append('@545')

        # 04-08-2022 если клаузулла безударная, то имеем кривое оформление ритмического рисунка.
        if not self.rhyming_tail.is_ok():
            self.total_score *= 0.1
            self.penalties.append('@550')

        # 06-08-2022 если в строке вообще ни одного ударения - это неприемлемо
        if sum((w.new_stress_pos != -1) for w in self.stressed_words) == 0:
            self.total_score *= 0.01
            self.penalties.append('@555')

        # добавка от 15-12-2021: два подряд ударных слога наказываем сильно!
        #if '11' in self.stress_signature_str:
        #    self.total_score *= 0.1
        #    self.penalties.append('@560')

        # 01-01-2022 ударную частицу "и" в начале строки наказываем сильно
        # 〚И́(0.500) споко́йно детворе́〛(0.500)
        if self.stressed_words[0].new_stress_pos == 1 and self.stressed_words[0].poetry_word.form.lower() == 'и':
            self.total_score *= 0.1
            self.penalties.append('@573')

        for word1, word2 in zip(self.stressed_words, self.stressed_words[1:]):
            # 28-12-2021 проверяем цепочки согласных в смежных словах
            n_adjacent_consonants = word1.poetry_word.trailing_consonants + word2.poetry_word.leading_consonants
            if n_adjacent_consonants > 5:
                self.total_score *= 0.5
                self.penalties.append('@309')

            # 01-01-2022 Штрафуем за ударный предлог перед существительным:
            # Все по́ дома́м - она и ра́да
            #     ^^^^^^^^
            if word1.poetry_word.upos == 'ADP' and word1.new_stress_pos > 0 and word2.poetry_word.upos in ('NOUN', 'PROPN') and word2.new_stress_pos > 0:
                self.total_score *= 0.5
                self.penalties.append('@317')

        for word1, word2, word3 in zip(self.stressed_words, self.stressed_words[1:], self.stressed_words[2:]):
            # 29-12-2021 Более двух подряд безударных слов - штрафуем
            if word1.new_stress_pos == -1 and word2.new_stress_pos == -1 and word3.new_stress_pos == -1:
                # 03.08.2022 Бывают цепочки из трех слов, среди которых есть частицы вообще без гласных:
                # я́ ж ведь не распла́чусь
                #   ^^^^^^^^^
                # Такие цепочки не штрафуем.
                #if count_vowels(word1.poetry_word.form) > 0 and count_vowels(word2.poetry_word.form) > 0 and count_vowels(word3.poetry_word.form) > 0:
                #    self.total_score *= 0.1
                #    self.penalties.append('@323')
                pass

        # 28-12-2021 штрафуем за подряд идущие короткие слова (1-2 буквы)
        #for word1, word2, word3 in zip(stressed_words, stressed_words[1:], stressed_words[2:]):
        #    if word1.is_short_word() and word2.is_short_word() and word3.is_short_word():
        #        self.total_score *= 0.2

        if sum(self.stress_signature) == 1:
            # Всего один ударный слог в строке с > 2 слогов... Очень странно.
            # 〚Что за недоразуме́нье〛
            # 00000010
            # 07.06.2022 Но если в строке всего одно слово или группа предлог+сущ - это нормально!
            if len(self.poetry_line.pwords) > 2 or (len(self.poetry_line.pwords) == 2 and self.poetry_line.pwords[-2].upos != 'ADP'):
                if sum(count_vowels(w.poetry_word.form) for w in self.stressed_words) > 2:
                    self.total_score *= 0.1
                    self.penalties.append('@335')
        else:
            # 01-01-2022 Детектируем разные сбои ритма
            #if self.stress_signature_str in aligner.bad_signature1:
            #    self.total_score *= 0.1
            #    self.penalties.append('@340')
            pass

        # Три безударных слога в конце - это очень странно:
        # Ходи́ть по то́нкому льду.		0101000
        if self.stress_signature_str.endswith('000'):
            self.total_score *= 0.1
            self.penalties.append('@626')

    def init_rhyming_tail(self):
        stressed_word = None
        unstressed_prefix = None
        unstressed_postfix_words = []

        # Ищем справа слово с ударением
        i = len(self.stressed_words)-1
        while i >= 0:
            pword = self.stressed_words[i]
            if pword.new_stress_pos != -1:  # or pword.poetry_word.n_vowels > 1:
                stressed_word = pword

                if re.match(r'^[аеёиоуыэюя]$', pword.poetry_word.form, flags=re.I) is not None:
                    # Ситуация, когда рифмуется однобуквенное слово, состоящее из гласной:
                    #
                    # хочу́ отшлё́пать анако́нду
                    # но непоня́тно по чему́
                    # вот у слона́ гора́здо ши́ре
                    # чем у                       <=======
                    if i > 0:
                        if re.match(r'\w', self.stressed_words[i-1].poetry_word.form[-1], flags=re.I):
                            unstressed_prefix = self.stressed_words[i-1].poetry_word.form[-1].lower()

                # все слова, кроме пунктуации, справа от данного сформируют безударный хвост клаузуллы
                for i2 in range(i+1, len(self.stressed_words)):
                    if self.stressed_words[i2].poetry_word.upos != 'PUNCT':
                        unstressed_postfix_words.append(self.stressed_words[i2])

                break
            i -= 1

        self.rhyming_tail = RhymingTail(unstressed_prefix, stressed_word, unstressed_postfix_words)

    def __repr__(self):
        if self.poetry_line is None or self.poetry_line.is_empty():
            return ''

        s = '〚' + ' '.join(w.__repr__() for w in self.stressed_words) + '〛'
        if self.total_score != 1.0:
            s += '(≈{:5.3f})'.format(self.total_score)
        return s

    def get_stressed_line(self, show_secondary_accentuation):
        s = ' '.join(w.get_stressed_form(show_secondary_accentuation) for w in self.stressed_words)
        s = normalize_whitespaces(s)
        return s

    def get_unstressed_line(self):
        s = self.get_stressed_line(False)
        s = s.replace('\u0301', '')
        return s

    def get_score(self):
        return self.total_score

    def score_sign1(self, etalog_sign, line_sign):
        if etalog_sign == 0 and line_sign == 1:
            # Ударный слог в безударном месте - сильный штраф
            return 0.1
        elif etalog_sign == 1 and line_sign == 0:
            # Безударный слог в ударном месте - небольшой штраф
            return 0.9
        else:
            # Ударности полностью соответствуют.
            return 1.0

    def map_meter(self, signature):
        l = len(signature)
        n = len(self.stress_signature) // l
        if (len(self.stress_signature) % l) > 0:
            n += 1

        expanded_sign = (signature * n)[:len(self.stress_signature)]

        sign_scores = [self.score_sign1(x, y) for x, y in zip(expanded_sign, self.stress_signature)]
        sign_score = reduce(lambda x, y: x * y, sign_scores)

        if sign_score < 1.0 and signature[0] == 1 and self.stress_signature[0] == 0:
            # Попробуем сместить вправо, добавив в начало один неударный слог.
            expanded_sign2 = (0,) + expanded_sign[:-1]
            sign_scores2 = [self.score_sign1(x, y) for x, y in zip(expanded_sign2, self.stress_signature)]
            sign_score2 = reduce(lambda x, y: x * y, sign_scores2)
            if sign_score2 > sign_score:
                return sign_score2

        return sign_score

    def split_to_syllables(self):
        output_tokens = []
        for word in self.stressed_words:
            if len(output_tokens) > 0:
                output_tokens.append('|')
            output_tokens.extend(word.split_to_syllables())
        return output_tokens

    #def get_last_rhyming_word(self):
    #    # Вернем последнее слово в строке, которое надо проверять на рифмовку.
    #    # Финальную пунктуацию игнорируем.
    #    for pword in self.stressed_words[::-1]:
    #        if pword.poetry_word.upos != 'PUNCT':
    #            return pword
    #    return None
    def get_rhyming_tail(self):
        return self.rhyming_tail


def count_vowels(s):
    return sum((c.lower() in 'уеыаоэёяию') for c in s)


def locate_Astress_pos(s):
    stress_pos = -1
    n_vowels = 0
    for c in s:
        if c.lower() in 'уеыаоэёяию':
            n_vowels += 1

        if c in 'АЕЁИОУЫЭЮЯ':
            stress_pos = n_vowels
            break

    return stress_pos


class PoetryLine(object):
    def __init__(self):
        self.text = None
        self.pwords = None

    def __len__(self):
        return len(self.pwords)

    def is_empty(self) -> bool:
        return not self.text

    @staticmethod
    def build(text, udpipe_parser, accentuator):
        poetry_line = PoetryLine()
        poetry_line.text = text
        poetry_line.pwords = []

        text2 = text

        # Отбиваем некоторые симполы пунктуации пробелами, чтобы они гарантировано не слиплись со словом
        # в токенизаторе UDPipe/Stanza.
        for c in '\'.‚,?!:;…-–—«»″”“„‘’`ʹ"˝[]‹›·<>*/=()+®©‛¨×№\u05f4':
            text2 = text2.replace(c, ' ' + c + ' ').replace('  ', ' ')

        parsings = udpipe_parser.parse_text(text2)
        if parsings is None:
            raise ValueError('Could not parse text: ' + text2)

        for parsing in parsings:
            # Если слово в строке всего одно, то частеречная разметка не будет нормально работать.
            # В этом случае мы просто берем все варианты ударения для этого единственного слова.
            nw = sum(t.upos != 'PUNCT' for t in parsing)
            if nw == 1:
                for ud_token in parsing:
                    word = ud_token.form.lower()

                    secondary_accentuation = accentuator.get_secondary_accentuation(word)

                    accentuations = []

                    if word in accentuator.ambiguous_accents2:
                        for stress_pos in accentuator.ambiguous_accents2[word]:
                            accentuations.append(WordAccentuation(stress_pos, secondary_accentuation))
                    elif word in accentuator.ambiguous_accents:
                        # Слово является омографом. Возьмем в работу все варианты ударения.
                        for stressed_form, tagsets in accentuator.ambiguous_accents[word].items():
                            stress_pos = locate_Astress_pos(stressed_form)
                            accentuations.append(WordAccentuation(stress_pos, secondary_accentuation))
                    else:
                        # 22.06.2022 для OOV делаем перебор всех вариантов ударности для одиночного слова в артишоках.
                        # При этом не используется функционал автокоррекции искажения и опечаток в ударяторе.
                        if word not in accentuator.word_accents_dict and any((c in word) for c in 'аеёиоуыэюя'):
                            n_vowels = 0
                            for c in word:
                                if c.lower() in 'уеыаоэёяию':
                                    n_vowels += 1
                                    stress_pos = n_vowels
                                    accentuations.append(WordAccentuation(stress_pos, secondary_accentuation))
                        else:
                            accentuations = accentuator.get_accents(word, ud_tags=ud_token.tags + [ud_token.upos])

                    pword = PoetryWord(ud_token.lemma, ud_token.form, ud_token.upos, ud_token.tags, accentuations)
                    poetry_line.pwords.append(pword)
            else:
                for ud_token in parsing:
                    word = ud_token.form.lower()

                    word_accentuations = accentuator.get_accents(word, ud_tags=ud_token.tags + [ud_token.upos])
                    #form2 = accentuator.yoficate(ud_token.form)
                    form2 = ud_token.form
                    #yoficated_form = accentuator.yoficate2(ud_token.form, ud_tags=ud_token.tags + [ud_token.upos])
                    pword = PoetryWord(lemma=ud_token.lemma,
                                       form=form2,
                                       upos=ud_token.upos,
                                       tags=ud_token.tags,
                                       accentuations=word_accentuations)

                    poetry_line.pwords.append(pword)

                    # stress_pos = accentuator.get_accent(word, ud_tags=ud_token.tags + [ud_token.upos])
                    #
                    # alt_stress_pos = []
                    # if count_vowels(word) > 0 and stress_pos == -1:
                    #     # Если слово допускает альтернативные ударения по списку, то берем первое из них (обычно это
                    #     # основное, самое частотное ударение), и не бросаем исключение, так как дальше матчер все равно
                    #     # будет перебирать все варианты по списку.
                    #     if word in accentuator.ambiguous_accents2:
                    #         px = accentuator.ambiguous_accents2[word]
                    #         stress_pos = px[0]
                    #     if stress_pos == -1:
                    #         # Мы можем оказаться тут из-за ошибки частеречной разметки. Проверим,
                    #         # входит ли слово в список омографов.
                    #         if word in accentuator.ambiguous_accents:
                    #             # Слово является омографом. Возьмем в работу все варианты ударения.
                    #
                    #             for stressed_form, tagsets in accentuator.ambiguous_accents[word].items():
                    #                 i = locate_Astress_pos(stressed_form)
                    #                 alt_stress_pos.append(i)
                    #             stress_pos = alt_stress_pos[0]
                    #
                    #         if stress_pos == -1:
                    #             msg = 'Could not locate stress position in word "{}"'.format(word)
                    #             raise ValueError(msg)
                    #
                    # form2 = accentuator.yoficate(ud_token.form)
                    # secondary_accentuation = accentuator.get_secondary_accentuation(word)
                    #
                    # pword = PoetryWord(ud_token.lemma, form2, ud_token.upos, ud_token.tags, stress_pos, alt_stress_pos, secondary_accentuation=secondary_accentuation)
                    #
                    # poetry_line.pwords.append(pword)

        poetry_line.locate_rhyming_word()
        return poetry_line

    @staticmethod
    def build_from_markup(markup_line, parser):
        pline = PoetryLine()
        pline.text = markup_line.replace('\u0301', '')
        pline.pwords = []

        text2 = markup_line
        for c in '.,?!:;…-–—«»”“„‘’`"':
            text2 = text2.replace(c, ' ' + c + ' ').replace('  ', ' ')

        # Надо поискать в исходной размеченной строке наше слово. Одинаковое слово может встретится 2 раза с разной
        # разметкой.
        line_spans = [(span.replace('\u0301', ''), span) for span in re.split(r'[.?,!:;…\-\s]', markup_line)]

        # удаляем расставленные ударения и выполняем полный анализ.
        parsings = parser.parse_text(text2.replace('\u0301', ''))

        for parsing in parsings:
            for ud_token in parsing:
                # определяем позицию гласного, помеченного как ударный.
                stress_pos = -1

                n_vowels = 0
                for ic, c in enumerate(ud_token.form):
                    if c.lower() in 'уеыаоэёяию':
                        # Нашли гласную. попробуем сделать ее ударной (поставить справа юникодный маркер ударения)
                        # и поискать в исходной разметке.
                        n_vowels += 1
                        needle = ud_token.form[:ic+1] + '\u0301' + ud_token.form[ic+1:]
                        # Ищем слева направо в сегментах с разметкой. Если слово найдется в сегменте, то его
                        # исключим из дальнейшего поиска, чтобы справится с ситуацией разного ударения в одинаковых словах.
                        for ispan, (span_form, span_markup) in enumerate(line_spans):
                            if span_form == ud_token.form and span_markup == needle:
                                stress_pos = n_vowels
                                line_spans = line_spans[:ispan] + line_spans[ispan+1:]
                                break

                        if stress_pos == -1:
                            # могут быть MWE с дефисом, которые распались на отдельные слова. Их ищем в исходной
                            # разметке целиком.
                            if re.search(r'\b' + needle.replace('-', '\\-').replace('.', '\\.') + r'\b', markup_line):
                                stress_pos = n_vowels
                                break

                        if stress_pos != -1:
                            break

                pword = PoetryWord(ud_token.lemma, ud_token.form, ud_token.upos, ud_token.tags, stress_pos)
                pline.pwords.append(pword)

        pline.locate_rhyming_word()
        return pline

    def locate_rhyming_word(self):
        # Отмечаем последнее слово в строке, так как оно должно ударяться, за исключением
        # очень редких случаев:
        # ... я же
        # ... ляжет
        if len(self.pwords) == 0:
            return

        located = False
        for pword in self.pwords[::-1]:
            if pword.upos != 'PUNCT':
                pword.is_rhyming_word = True
                located = True
                break
        if not located:
            msg = 'Could not locate rhyming word in line ' + self.text
            raise ValueError(msg)

    def __repr__(self):
        return ' '.join([pword.__repr__() for pword in self.pwords])

    def get_num_syllables(self):
        return sum(word.n_vowels for word in self.pwords)

    def get_first_stress_variants(self, aligner):
        swords = [pword.get_first_stress_variant(aligner) for pword in self.pwords]
        return LineStressVariant(self, swords, aligner)


class PoetryAlignment(object):
    def __init__(self, poetry_lines, score, meter, rhyme_scheme, rhyme_graph, metre_mappings, inner_alignments=None):
        self.poetry_lines = poetry_lines
        self.score = score

        self.metre_mappings = metre_mappings
        if meter == 'dolnik':
            self.meter = meter
        elif metre_mappings:
            self.meter = metre_mappings[0].get_canonic_meter()
        else:
            self.meter = meter

        self.rhyme_scheme = rhyme_scheme
        self.rhyme_graph = rhyme_graph
        self.error_text = None
        self.inner_alignments = inner_alignments

    def __repr__(self):
        s = '{} {}({:5.3f}):\n'.format(self.meter, self.rhyme_scheme, self.score)
        s += '\n'.join(map(str, self.poetry_lines))
        return s

    def is_poetry(self) -> bool:
        return self.meter and self.score > 0.0

    @staticmethod
    def build_n4(alignments4, total_score):
        if len(alignments4) == 1:
            return alignments4[0]

        poetry_lines = []
        rhyme_graph = []

        for i, a in enumerate(alignments4):
            if i > 0:
                # добавляем пустую строку-разделитель
                poetry_lines.append(LineStressVariant.build_empty_line())
                rhyme_graph.append(None)

            poetry_lines.extend(a.poetry_lines)
            rhyme_graph.extend(a.rhyme_graph)

        rhyme_scheme = None
        if all(isinstance(a.rhyme_scheme, str) for a in alignments4):
            rhyme_scheme = ' '.join(a.rhyme_scheme for a in alignments4)

        return PoetryAlignment(poetry_lines,
                               total_score,
                               alignments4[0].meter,
                               rhyme_scheme=rhyme_scheme,
                               rhyme_graph=rhyme_graph,
                               metre_mappings=list(itertools.chain(*(a.metre_mappings for a in alignments4))),
                               inner_alignments=alignments4)

    @staticmethod
    def build_no_rhyming_result(poetry_lines):
        a = PoetryAlignment(poetry_lines, 0.0, None, None, metre_mappings=None, rhyme_graph=None)
        a.error_text = 'Отсутствует рифмовка последних слов'
        return a

    def get_num_rhymes(self):
        if self.rhyme_graph:
            return sum((j is not None) for j in self.rhyme_graph)
        else:
            return 0

    def get_stressed_lines(self, show_secondary_accentuation):
        if show_secondary_accentuation:
            lines = []
            imetre = 0
            for poetry_line in self.poetry_lines:
                if poetry_line.is_empty():
                    lines.append('')
                else:
                    if self.metre_mappings[imetre].count_stress_marks() == 0:
                        lines.append(poetry_line.get_stressed_line(show_secondary_accentuation))
                    else:
                        lines.append(self.metre_mappings[imetre].get_stressed_line())
                    imetre += 1

            return '\n'.join(lines)
        else:
            return '\n'.join(x.get_stressed_line(show_secondary_accentuation=False) for x in self.poetry_lines)

    def get_unstressed_lines(self):
        return '\n'.join(x.get_unstressed_line() for x in self.poetry_lines)

    def split_to_syllables(self, show_secondary_accentuation, do_arabize):
        lx = []

        imapping = 0
        for pline in self.poetry_lines:
            if pline.is_empty():
                lx.append('')
            else:
                line_mapping = self.metre_mappings[imapping]

                line_syllables = []

                for word_mapping in line_mapping.word_mappings:
                    syllables_with_accentuation = []
                    stress_signature = word_mapping.get_stress_signature_str()
                    if len(stress_signature) == 0:
                        syllables_with_accentuation.append(word_mapping.word.form)
                    else:
                        sx = get_syllables(word_mapping.word.form)

                        for i, syllable in enumerate(sx):
                            syllable_text = syllable.text
                            if stress_signature[i] == '1':
                                syllable_text = re.sub(r'([аеёиоуыэюя])', '\\1\u0301', syllable_text)
                            elif stress_signature[i] == '2':
                                syllable_text = re.sub(r'([аеёиоуыэюя])', '\\1\u0300', syllable_text)

                            syllables_with_accentuation.append(syllable_text)

                    if line_syllables:
                        line_syllables.append('|')  # word separator
                    line_syllables.extend(syllables_with_accentuation)

                if do_arabize:
                    line_syllables = line_syllables[::-1]

                lx.append(' '.join(line_syllables))

                imapping += 1

        return lx

    def get_line_text(self, line_index):
        pline = self.poetry_lines[line_index].poetry_line
        return pline.text if pline is not None and pline.text is not None else ''


# Мы проверяем только эти 5 вариантов чередования ударных и безударных слогов.
# Более сложные случаи отбрасываем, они слишком тяжелы для восприятия.
meters = [('ямб', (0, 1)),
          ('хорей', (1, 0)),
          ('дактиль', (1, 0, 0)),
          ('амфибрахий', (0, 1, 0)),
          ('анапест', (0, 0, 1))]


class CollocationStress(object):
    def __init__(self):
        self.words = []
        self.stressed_word_index = -1
        self.stress_pos = -1

    def __repr__(self):
        return ' '.join(self.words) if self.words else ''

    def __len__(self):
        return len(self.words)

    def key(self):
        return tuple(self.words)

    @staticmethod
    def load_collocation(colloc_str):
        res = CollocationStress()
        words = colloc_str.split()
        res.words = [w.lower() for w in words]
        uvx = 'АЕЁИОУЫЭЮЯ'
        vx = 'аеёиоуыэюя'
        for iword, word in enumerate(words):
            if any((c in word) for c in uvx):
                res.stressed_word_index = iword
                vowel_count = 0
                for c in word:
                    if c.lower() in vx:
                        vowel_count += 1
                        if c in uvx:
                            res.stress_pos = vowel_count
                            break

        return res



class SongAlignmentBlock(object):
    """Структура для хранения результатов разметки куплета или припева в песне"""
    def __init__(self, title, alignment):
        self.title = title  # пусто, или "Припев:", или "Припев."
        self.alignment = alignment

    def get_total_score(self):
        return self.alignment.score if self.alignment is not None else 0.0

    def __repr__(self):
        chunks = []
        if self.title:
            chunks.append(self.title)
        if self.alignment:
            chunks.append(self.alignment.get_stressed_lines())
        return '\n'.join(chunks)

    def get_markup(self):
        lines = []
        if self.title:
            lines.append(self.title)
        if self.alignment:
            lines.append(self.alignment.get_stressed_lines(show_secondary_accentuation=False))
        return '\n'.join(lines)

    def get_rhyming_ratio(self):
        if self.alignment is None:
            return 0, 0
        else:
            return len(self.alignment.poetry_lines), self.alignment.get_num_rhymes()

    def get_syllabized(self):
        chunks = []
        if self.title:
            swords = []
            for word in re.sub(r'([.:,])', ' \\1', self.title).split(' '):
                sx = get_syllables(word)
                swords.append(' '.join(s.text for s in sx))

            chunks.append(' | '.join(swords))
            chunks.append(' | \n | ')
        if self.alignment:
            chunks.append(' | \n | '.join(self.alignment.split_to_syllables(show_secondary_accentuation=False, do_arabize=False)))

        return ''.join(chunks)


class SongAlignment(object):
    """Структура для хранения результатов разметки текста песни"""
    def __init__(self):
        self.blocks = []

    def get_total_score(self):
        if self.blocks:
            return min(block.get_total_score() for block in self.blocks)
        else:
            return 0.0

    def __repr__(self):
        if self.blocks:
            return '\n\n'.join(map(str, self.blocks))
        else:
            return '<<< EMPTY SONG MARKUP >>>'

    def get_markup(self):
        return '\n\n'.join(block.get_markup() for block in self.blocks)

    def get_syllabized(self):
        return ' | \n | \n | '.join(block.get_syllabized() for block in self.blocks)

    def get_rhyming_rate(self):
        num_lines = 0
        num_rhymes = 0
        for block in self.blocks:
            l, r = block.get_rhyming_ratio()
            num_lines += l
            num_rhymes += r
        return num_rhymes / max((num_lines - 1 + 1e-6), 1e-6)


class WordSegmentation(object):
    def __init__(self, segments):
        self.segments = segments
        self.without_prefix = ''.join(s.split(':')[0] for s in segments if not s.endswith('PREF'))

    def __repr__(self):
        return ' '.join(self.segments)


# https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)


class RhymeGraphNode(object):
    def __init__(self):
        self.offset_to_right = 0
        self.fit_to_right = False
        self.offset_to_left = 0
        self.fit_to_left = True
        self.rhyme_scheme_letter = '-'


def convert_rhyme_graph_to_scheme(rhyme_graph: List[int]) -> str:
    line_rhymes = [RhymeGraphNode() for _ in rhyme_graph]
    for i1, offset in enumerate(rhyme_graph):
        if offset is not None:
            i2 = i1 + offset
            line_rhymes[i1].offset_to_right = (i2-i1)
            line_rhymes[i1].fit_to_right = True
            line_rhymes[i2].offset_to_left = (i1-i2)
            line_rhymes[i2].fit_to_left = True

    current_char = ord('A')
    for i, rhyme in enumerate(line_rhymes):
        if rhyme.offset_to_right > 0:
            if rhyme.rhyme_scheme_letter == '-':
                rhyme.rhyme_scheme_letter = chr(current_char)
                current_char += 1

            line_rhymes[i+rhyme.offset_to_right].rhyme_scheme_letter = rhyme.rhyme_scheme_letter

    rhyme_scheme = ''.join([rhyme.rhyme_scheme_letter for rhyme in line_rhymes])

    return rhyme_scheme


class PoetryStressAligner(object):
    def __init__(self, udpipe, accentuator, model_dir: str, enable_dolnik: bool=True):
        self.max_words_per_line = 10
        self.early_stopping_threshold_score = 0.7
        self.enable_dolnik = enable_dolnik
        self.udpipe = udpipe
        self.accentuator = accentuator
        self.allow_stress_guessing_for_oov = False
        self.allow_fuzzy_rhyming = True

        with open(os.path.join(model_dir, 'scansion_tool.pkl'), 'rb') as f:
            self.collocations = CustomUnpickler(f).load()
            self.word_segmentation = CustomUnpickler(f).load()

    @staticmethod
    def compile(data_dir, output_dir):
        collocations = collections.defaultdict(list)

        with io.open(os.path.join(data_dir, 'poetry/dict/collocation_accents.dat'), 'r', encoding='utf-8') as rdr:
            for line in rdr:
                line = line.strip()
                if line.startswith('#') or len(line) == 0:
                    continue

                c = CollocationStress.load_collocation(line)

                collocations[c.key()].append(c)

        word_segmentation = dict()
        with io.open(os.path.join(data_dir, 'poetry/dict/word_segmentation.csv'), 'r', encoding='utf-8') as rdr:
            header = rdr.readline()
            for line in rdr:
                fields = line.strip().split(',')
                word = fields[0]
                segments = fields[1].split('/')
                word_segmentation[word] = WordSegmentation(segments)

        with open(os.path.join(output_dir, 'scansion_tool.pkl'), 'wb') as f:
            pickle.dump(collocations, f)
            pickle.dump(word_segmentation, f)

    def map_meter(self, signature, lines):
        scores = [line.map_meter(signature) for line in lines]
        return reduce(lambda x, y: x*y, scores)

    def map_meters(self, lines):
        best_score = -1.0
        best_meter = None
        for name, signature in meters:
            score = self.map_meter(signature, lines)
            if score > best_score:
                best_score = score
                best_meter = name
        return best_meter, best_score

    def align(self, lines0, check_rhymes=True, check_poor_poetry=True):
        # Иногда для наглядности можем выводить сгенерированные стихи вместе со значками ударения.
        # Эти значки мешают работе алгоритма транскриптора, поэтому уберем их сейчас.
        lines = [line.replace('\u0301', '').replace('\u0300', '') for line in lines0]
        nlines = len(lines)
        if nlines == 1:
            return self.align1(lines)
        elif nlines == 2:
            return self.align2(lines, check_rhymes)
        elif nlines == 4:
            return self.align4(lines, check_rhymes)
        else:
            return self.align_nx(lines, check_rhymes, check_poor_poetry)

    def align_nx(self, lines, check_rhymes, check_poor_poetry=True):
        total_score = 1.0
        block = []
        block_alignments = []

        for line in lines+['']:
            if line:
                block.append(line)
            else:
                if block:
                    if len(block) == 4:
                        alignment = self.align4(block, check_rhymes=check_rhymes)
                    elif len(block) == 2:
                        alignment = self.align2(block, check_rhymes=check_rhymes)
                    else:
                        alignment = self.align_nonstandard_block(block, check_rhymes=check_rhymes)

                    if alignment is None:
                        total_score = 0.0
                        break

                    block_score = alignment.score
                    if check_poor_poetry and self.detect_poor_poetry(alignment):
                        block_score *= 0.1

                    # TODO: штрафовать за разные виды метра в блоках?

                    #total_score *= block_score
                    total_score = min(total_score, block_score)
                    block_alignments.append(alignment)

                    block = []

        result = PoetryAlignment.build_n4(block_alignments, total_score)

        if [len(block.poetry_lines) for block in block_alignments] == [4, 4, 3, 3]:
            # Возможно, это сонет в формате 4+4+3+3
            # Проверим кросс-рифмовку и скорректируем общую схему рифмовки.
            if [block.rhyme_scheme for block in block_alignments] == ['ABBA', 'ABBA', 'A-A', 'A-A']:
                rhyme_1_4 = block_alignments[0].poetry_lines[3].get_rhyming_tail()
                rhyme_2_1 = block_alignments[1].poetry_lines[0].get_rhyming_tail()
                r_14_21 = self.check_rhyming(rhyme_1_4, rhyme_2_1)

                rhyme_1_3 = block_alignments[0].poetry_lines[2].get_rhyming_tail()
                rhyme_2_2 = block_alignments[1].poetry_lines[1].get_rhyming_tail()
                r_13_22 = self.check_rhyming(rhyme_1_3, rhyme_2_2)

                if r_14_21 and r_13_22:
                    rhyme_3_3 = block_alignments[2].poetry_lines[2].get_rhyming_tail()
                    rhyme_4_1 = block_alignments[3].poetry_lines[0].get_rhyming_tail()
                    r_33_41 = self.check_rhyming(rhyme_3_3, rhyme_4_1)
                    if r_33_41:
                        rhyme_3_2 = block_alignments[2].poetry_lines[1].get_rhyming_tail()
                        rhyme_4_2 = block_alignments[3].poetry_lines[1].get_rhyming_tail()
                        r_32_42 = self.check_rhyming(rhyme_3_2, rhyme_4_2)
                        if r_32_42:
                            new_rhyme_scheme = 'ABBA ABBA CDC CDC'
                            result.rhyme_scheme = new_rhyme_scheme
                            # TODO обновить result.rhyme_graf

        return result

    def check_rhyming(self, rhyming_tail1, rhyming_tail2) -> bool:
        if not rhyming_tail1.is_ok() or not rhyming_tail2.is_ok():
            return False

        poetry_word1 = rhyming_tail1.stressed_word
        poetry_word2 = rhyming_tail2.stressed_word

        if rhyming_tail1.is_simple() and rhyming_tail2.is_simple():
            # 01.02.2022 проверяем слова с ударениями по справочнику рифмовки
            f1 = poetry_word1.stressed_form
            f2 = poetry_word2.stressed_form
            if (f1, f2) in self.accentuator.rhymed_words or (f2, f1) in self.accentuator.rhymed_words:
                return True

            # Считаем, что слово не рифмуется само с собой
            # 09-08-2022 отключил эту проверку, так как в is_poor_rhyming делает более качественная проверка!
            #if poetry_word1.form.lower() == poetry_word2.form.lower():
            #    return False

        unstressed_tail1 = rhyming_tail1.unstressed_tail
        unstressed_tail2 = rhyming_tail2.unstressed_tail

        r = rhymed2(self.accentuator,
                    poetry_word1.poetry_word.form, poetry_word1.new_stress_pos, [poetry_word1.poetry_word.upos] + poetry_word1.poetry_word.tags, rhyming_tail1.prefix, unstressed_tail1,
                    poetry_word2.poetry_word.form, poetry_word2.new_stress_pos, [poetry_word2.poetry_word.upos] + poetry_word2.poetry_word.tags, rhyming_tail2.prefix, unstressed_tail2)
        if r:
            return True

        if self.allow_fuzzy_rhyming:
            return rhymed_fuzzy2(self.accentuator,
                                poetry_word1.poetry_word.form, poetry_word1.new_stress_pos, [poetry_word1.poetry_word.upos] + poetry_word1.poetry_word.tags, rhyming_tail1.prefix, unstressed_tail1,
                                poetry_word2.poetry_word.form, poetry_word2.new_stress_pos, [poetry_word2.poetry_word.upos] + poetry_word2.poetry_word.tags, rhyming_tail2.prefix, unstressed_tail2)

        return False

    def align_AABA(self, lines):
        res = self.align4(lines, check_rhymes=True)
        if res.rhyme_scheme == 'AABA':
            return res
        else:
            #plines = [PoetryLine.build(line, self.udpipe, self.accentuator) for line in lines]
            #return PoetryAlignment.build_no_rhyming_result([pline.get_stress_variants(self)[0] for pline in plines])
            return None

    def detect_stress_lacunas(self, line_mappings) -> bool:
        for line_mapping in line_mappings:
            if self.detect_stress_lacunas1(line_mapping):
                return True
        return False

    def detect_stress_lacunas1(self, line_mapping) -> bool:
        nwords = len(line_mapping.word_mappings)
        for iword, word_mapping in enumerate(line_mapping.word_mappings):
            if (len(word_mapping.syllabic_mapping) > 1 and word_mapping.word.poetry_word.upos in ('NOUN', 'ADJ', 'ADV', 'VERB')) \
                 or (len(word_mapping.syllabic_mapping) == 1 and word_mapping.word.poetry_word.upos in ('NOUN', 'ADJ', 'VERB') and line_mapping.get_score() < 0.40):
                if 'TP' not in word_mapping.syllabic_mapping:  # ударений нет
                    # Проверим, что слово не участвует в словосочетании с переносом ударения на клитику, например "по лесу".
                    this_word = word_mapping.word.form.lower()
                    prev_word = None
                    next_word = None

                    prosodic_collocation_found = False

                    if iword > 0:
                        # Проверим сочетание с предыдущим или последующим словом для словосочетаний типа "по лесу"
                        prev_word = line_mapping.word_mappings[iword - 1].word.form.lower()

                        key = (prev_word, this_word)
                        if key in self.collocations:
                            prosodic_collocation_found = True

                    if iword < nwords - 1:
                        # Проверим сочетания в контексте из 3х слов для словосочетаний типа "бок о бок"
                        next_word = line_mapping.word_mappings[iword + 1].word.form.lower()

                        key = (this_word, next_word)
                        if key in self.collocations:
                            prosodic_collocation_found = True

                    if prev_word and next_word:
                        # Проверим сочетания в контексте из 3х слов для словосочетаний типа "бок о бок"
                        key = (prev_word, this_word, next_word)
                        if key in self.collocations:
                            prosodic_collocation_found = True

                    if iword > 1:
                        prev2_word = line_mapping.word_mappings[iword - 2].word.form.lower()
                        key = (prev2_word, prev_word, this_word)
                        if key in self.collocations:
                            prosodic_collocation_found = True

                    if iword < nwords - 2:
                        next2_word = line_mapping.word_mappings[iword + 2].word.form.lower()
                        key = (this_word, next_word, next2_word)
                        if key in self.collocations:
                            prosodic_collocation_found = True

                    if not prosodic_collocation_found:
                        # Unstressed word defect detected.
                        return True
        return False

    def align1(self, lines):
        """Разметка однострочника."""
        tokens = tokenize(lines[0])
        ntokens = len(tokens)
        if ntokens > self.max_words_per_line:
            raise ValueError(
                'Line is too long @1655: max_words_per_line={} tokens={} line="{}"'.format(self.max_words_per_line, '|'.join(tokens), lines[0]))

        pline1 = PoetryLine.build(lines[0], self.udpipe, self.accentuator)

        # 08.11.2022 добавлена защита от взрыва числа переборов для очень плохих генераций.
        if sum((pword.n_vowels >= 1) for pword in pline1.pwords) >= self.max_words_per_line:
            raise ValueError('Line is too long @1652: "{}"'.format(pline1))

        best_score = 0.0
        best_metre_name = None
        best_mapping = None

        for allow_stress_shift in [False, True]:
            # Заранее сгенерируем для каждого слова варианты спеллчека и ударения...
            # stressed_words_groups = [pword.get_stress_variants(self, allow_stress_shift=True) for pword in pline1.pwords]
            stressed_words_chain = StressVariantsSlot.build(poetry_words=pline1.pwords, aligner=self, allow_stress_shift=allow_stress_shift, allow_unstress12=True)

            for metre_name, metre_signature in meters:
                cursor = MetreMappingCursor(metre_signature, prefix=0)
                for metre_mapping in cursor.map(stressed_words_chain, self):
                    if metre_mapping.get_score() > best_score:
                        best_score = metre_mapping.get_score()
                        best_metre_name = metre_name
                        best_mapping = metre_mapping
            if best_score > 0.4:
                break

        # 13.04.2025 Trying to apply dolnik patterns if not succeeded with regular meters
        if best_score <= self.early_stopping_threshold_score and self.enable_dolnik:
            # Определим максимальное число слогов. Это нужно для дольников.
            num_syllables = [pline1.get_num_syllables()]
            dolnik_patterns = self.get_dolnik_patterns(num_syllables)
            stressed_words_groups0 = [StressVariantsSlot.build(poetry_words=pline1.pwords, aligner=self, allow_stress_shift=False, allow_unstress12=True)]

            for dolnik_pattern in dolnik_patterns:
                new_stress_lines = []
                pline = pline1
                cursor = MetreMappingCursor(dolnik_pattern[0], prefix=0)

                for metre_mapping in cursor.map(stressed_words_groups0[0], self):
                    if metre_mapping.score > best_score:
                        stressed_words = [m.word for m in metre_mapping.word_mappings]
                        new_stress_line = LineStressVariant(pline, stressed_words, self)
                        new_stress_lines.append((best_mapping, new_stress_line))

                        best_score = metre_mapping.score
                        best_metre_name = 'dolnik'
                        best_mapping = metre_mapping


        # Возвращаем найденный лучший вариант разметки и его оценку
        do_fallback_proza_accentuation = False
        new_stress_line = None

        # 16.04.2025 If the score is below the threshold, then we are dealing not with one-line poetry, but with prose.
        # Let's roll back to the prose markup.
        if best_score < 0.15:
            do_fallback_proza_accentuation = True
        else:
            # 18.04.2025 для шуточной поговорки "мал, да уд ал", получаем безударное последнее слово.
            # Пробуем прозаическую разметку, и если она дает внутреннюю рифму - принимаем прозаическую разметку.
            last_word_mapping = None
            for word_mapping in best_mapping.word_mappings[::-1]:
                if len(word_mapping.syllabic_mapping) > 0:
                    last_word_mapping = word_mapping
                    break

            if last_word_mapping and 'TP' not in last_word_mapping.syllabic_mapping:
                new_stress_line = pline1.get_first_stress_variants(self)
                rhyme_word = new_stress_line.get_rhyming_tail()
                if rhyme_word.is_ok() and rhyme_word.is_simple():
                    # Поищем рифму к этому слову внутри.
                    for inner_word in new_stress_line.stressed_words[0:-1]:
                        if inner_word.poetry_word.form != last_word_mapping.word.form:
                            inner_rhyme = RhymingTail(unstressed_prefix=None, stressed_word=inner_word,
                                                      unstressed_postfix_words=[])
                            if self.check_rhyming(inner_rhyme, rhyme_word):
                                # Internal rhyme is found
                                do_fallback_proza_accentuation = True
                                break

            if not do_fallback_proza_accentuation:
                # 17.04.2024 Для поговорки "Не страшна врагов туча, если армия могуча." при подгонке метра получаем
                # разметку с безударным существительным "врагов".
                # В таких случаях разумно откатиться на прозаическую разметку.

                if self.detect_stress_lacunas1(best_mapping):
                    do_fallback_proza_accentuation = True

                # nwords = len(best_mapping.word_mappings)
                # for iword, word_mapping in enumerate(best_mapping.word_mappings):
                #     if (len(word_mapping.syllabic_mapping) > 1 and word_mapping.word.poetry_word.upos in ('NOUN', 'ADJ', 'ADV', 'VERB'))\
                #             or (len(word_mapping.syllabic_mapping)==1 and word_mapping.word.poetry_word.upos in ('NOUN', 'ADJ', 'VERB') and best_score < 0.40):
                #         if 'TP' not in word_mapping.syllabic_mapping:  # ударений нет
                #             # Проверим, что слово не участвует в словосочетании с переносом ударения на клитику, например "по лесу".
                #             this_word = word_mapping.word.form.lower()
                #             prev_word = None
                #             next_word = None
                #
                #             prosodic_collocation_found = False
                #
                #             if iword > 0:
                #                 # Проверим сочетание с предыдущим или последующим словом для словосочетаний типа "по лесу"
                #                 prev_word = best_mapping.word_mappings[iword-1].word.form.lower()
                #
                #                 key = (prev_word, this_word)
                #                 if key in self.collocations:
                #                     prosodic_collocation_found = True
                #
                #             if iword < nwords-1:
                #                 # Проверим сочетания в контексте из 3х слов для словосочетаний типа "бок о бок"
                #                 next_word = best_mapping.word_mappings[iword+1].word.form.lower()
                #
                #                 key = (this_word, next_word)
                #                 if key in self.collocations:
                #                     prosodic_collocation_found = True
                #
                #             if prev_word and next_word:
                #                 # Проверим сочетания в контексте из 3х слов для словосочетаний типа "бок о бок"
                #                 key = (prev_word, this_word, next_word)
                #                 if key in self.collocations:
                #                     prosodic_collocation_found = True
                #
                #             if iword > 1:
                #                 prev2_word = best_mapping.word_mappings[iword-2].word.form.lower()
                #                 key = (prev2_word, prev_word, this_word)
                #                 if key in self.collocations:
                #                     prosodic_collocation_found = True
                #
                #             if iword < nwords-2:
                #                 next2_word = best_mapping.word_mappings[iword+2].word.form.lower()
                #                 key = (this_word, next_word, next2_word)
                #                 if key in self.collocations:
                #                     prosodic_collocation_found = True
                #
                #             if not prosodic_collocation_found:
                #                 do_fallback_proza_accentuation = True
                #                 break

            if not do_fallback_proza_accentuation:
                new_stress_line = pline1.get_first_stress_variants(self)

                rhyme_word = new_stress_line.get_rhyming_tail()
                if rhyme_word.is_ok() and rhyme_word.is_simple():
                    # Поищем рифму к этому слову внутри.
                    for inner_word in new_stress_line.stressed_words[0:-1]:
                        if inner_word.poetry_word.form != rhyme_word.stressed_word.poetry_word.form:
                            inner_rhyme = RhymingTail(unstressed_prefix=None, stressed_word=inner_word, unstressed_postfix_words=[])
                            if self.check_rhyming(inner_rhyme, rhyme_word):
                                do_fallback_proza_accentuation = True
                                break

        if do_fallback_proza_accentuation:
            if new_stress_line is None:
                new_stress_line = pline1.get_first_stress_variants(self)

            best_mapping = MetreMappingResult.build_from_nonpoetry(new_stress_line)

            best_score = 0.0
            best_metre_name = None
        else:
            stressed_words = [m.word for m in best_mapping.word_mappings]
            new_stress_line = LineStressVariant(pline1, stressed_words, self)

        best_variant = [new_stress_line]

        return PoetryAlignment(best_variant, best_score, best_metre_name, rhyme_scheme='', rhyme_graph=[None], metre_mappings=[best_mapping])

    def get_prefixes_for_meter(selfself, metre_signature):
        return [0] if len(metre_signature)==2 else [0, 1]

    def align2(self, lines, check_rhymes):
        """ Разметка двустрочника """
        plines = [PoetryLine.build(line, self.udpipe, self.accentuator) for line in lines]

        # 08.11.2022 добавлена защита от взрыва числа переборов для очень плохих генераций.
        for pline in plines:
            if sum((pword.n_vowels>=1) for pword in pline.pwords) >= self.max_words_per_line:
                raise ValueError('Line is too long @1688: "{}"'.format(pline))

        rhyming_detection_cache = dict()

        best_score = 0.0
        best_metre = None
        best_rhyme_scheme = None
        best_rhyme_graph = None
        best_variant = None

        for allow_stress_shift in [True,]:  #[False, True]:
            if best_score > self.early_stopping_threshold_score:
                break

            # Заранее сгенерируем для каждого слова варианты спеллчека и ударения...
            stressed_words_groups = [StressVariantsSlot.build(poetry_words=pline.pwords, aligner=self, allow_stress_shift=allow_stress_shift, allow_unstress12=True) for pline in plines]

            # Для каждой строки перебираем варианты разметки и оставляем по 2 варианта в каждом метре.
            for metre_name, metre_signature in meters:
                best_scores = dict()

                # В каждой строке перебираем варианты расстановки ударений.
                for ipline, pline in enumerate(plines):
                    best_scores[ipline] = dict()

                    for prefix in self.get_prefixes_for_meter(metre_signature):
                        cursor = MetreMappingCursor(metre_signature, prefix=prefix)
                        metre_mappings = cursor.map(stressed_words_groups[ipline], self)

                        for metre_mapping in metre_mappings[:2]:  # берем только 2(?) лучших варианта
                            stressed_words = [m.word for m in metre_mapping.word_mappings]
                            new_stress_line = LineStressVariant(pline, stressed_words, self)

                            if new_stress_line.get_rhyming_tail().is_ok():
                                tail_str = new_stress_line.get_rhyming_tail().__repr__()
                                score = metre_mapping.get_score()
                                if tail_str not in best_scores[ipline]:
                                    prev_score = -1e3
                                else:
                                    prev_score = best_scores[ipline][tail_str][0].get_score()
                                if score > prev_score:
                                    best_scores[ipline][tail_str] = (metre_mapping, new_stress_line)

                # Теперь для каждой исходной строки имеем несколько вариантов расстановки ударений.
                # Перебираем сочетания этих вариантов, проверяем рифмовку и оставляем лучший вариант для данного метра.
                stressed_lines2 = [list() for _ in range(2)]
                for iline, items2 in best_scores.items():
                    stressed_lines2[iline].extend(items2.values())

                vvx = list(itertools.product(*stressed_lines2))
                for ivar, plinev in enumerate(vvx):
                    # plinev это набор из двух экземпляров кортежей (MetreMappingResult, LineStressVariant).

                    # Определяем рифмуемость
                    rhyme_scheme = None
                    rhyme_score = 1.0

                    last_pwords = [pline[1].get_rhyming_tail() for pline in plinev]
                    if self.check_rhyming(last_pwords[0], last_pwords[1]):
                        rhyme_scheme = 'AA'
                        rhyme_graph = [1, None]
                    else:
                        rhyme_scheme = '--'
                        rhyme_score = 0.75
                        rhyme_graph = [None, None]

                    # Если две строки полностью совпадают в позициях ударений, то дисконтируем отличие техничности от 1.
                    # Например:
                    #
                    # Жале́я ми́р, земле́ не предава́й
                    # Гряду́щих ле́т прекра́сный урожа́й!
                    #
                    # 0101010001
                    # 0101010001
                    line_tscores = self.recalc_line_tscores(plinev, rhyme_graph)
                    #line_tscores = [pline[0].get_score() for pline in plinev]
                    #if plinev[0][1].stress_signature == plinev[1][1].stress_signature:
                    #    for i in range(2):
                    #        line_tscores[i] += (1.0 - line_tscores[i])*0.50

                    total_score = rhyme_score * mul(line_tscores)
                    if total_score > best_score:
                        best_score = total_score
                        best_metre = metre_name
                        best_rhyme_scheme = rhyme_scheme
                        best_variant = plinev
                        best_rhyme_graph = rhyme_graph

                if best_score > self.early_stopping_threshold_score:
                    break

        # Если не получилось найти хороший маппинг с основными метрами, то пробуем дольники
        if best_score <= self.early_stopping_threshold_score and self.enable_dolnik:
            # Определим максимальное число слогов. Это нужно для дольников.
            num_syllables = [pline.get_num_syllables() for pline in plines]
            dolnik_patterns = self.get_dolnik_patterns(num_syllables)
            stressed_words_groups0 = [
                StressVariantsSlot.build(poetry_words=pline.pwords, aligner=self, allow_stress_shift=False, allow_unstress12=True) for
                pline in plines]

            for dolnik_pattern in dolnik_patterns:
                new_stress_lines = []
                for ipline, pline in enumerate(plines):
                    cursor = MetreMappingCursor(dolnik_pattern[ipline % len(dolnik_pattern)], prefix=0)

                    new_stress_line = None
                    best_mapping = None
                    max_score = 0.0

                    for metre_mapping in cursor.map(stressed_words_groups0[ipline], self):
                        if metre_mapping.score > max_score:
                            max_score = metre_mapping.score
                            stressed_words = [m.word for m in metre_mapping.word_mappings]
                            new_stress_line = LineStressVariant(pline, stressed_words, self)
                            best_mapping = metre_mapping

                    new_stress_lines.append((best_mapping, new_stress_line))

                # Определяем рифмуемость
                last_pwords = [line[1].get_rhyming_tail() for line in new_stress_lines]
                rhyme_scheme, rhyme_score, rhyme_graph = self.detect_rhyming(last_pwords, rhyming_detection_cache)

                total_score = rhyme_score * mul([pline[0].get_score() for pline in new_stress_lines])
                if total_score > best_score:
                    best_score = total_score
                    best_metre = 'dolnik'
                    best_rhyme_scheme = rhyme_scheme
                    best_variant = new_stress_lines
                    best_rhyme_graph = rhyme_graph


        if best_variant is None:
            # В этом случае вернем результат с нулевым скором и особым текстом, чтобы
            # можно было вывести в лог строки с каким-то дефолтными
            return PoetryAlignment.build_no_rhyming_result([pline.get_first_stress_variants(self) for pline in plines])
        else:
            # Возвращаем найденный вариант разметки и его оценку
            best_lines = [v[1] for v in best_variant]
            metre_mappings = [v[0] for v in best_variant]
            return PoetryAlignment(best_lines, best_score, best_metre,
                                   rhyme_scheme=best_rhyme_scheme,
                                   rhyme_graph=best_rhyme_graph,
                                   metre_mappings=metre_mappings)

    def detect_rhyming(self, last_pwords, rhyming_detection_cache):
        k = tuple(map(str, last_pwords))
        if k in rhyming_detection_cache:
            return rhyming_detection_cache[k]

        rhyme_scheme, rhyme_score, rhyme_graph = self.detect_rhyming_core(last_pwords)
        rhyming_detection_cache[k] = (rhyme_scheme, rhyme_score, rhyme_graph)
        return rhyme_scheme, rhyme_score, rhyme_graph

    def detect_rhyming_core(self, last_pwords):
        rx = dict()
        rhyme_graph = []
        for i in range(len(last_pwords) - 1):
            i_rhyming = None
            for j in range(i + 1, len(last_pwords)):
                r = self.check_rhyming(last_pwords[i], last_pwords[j])
                rx[(i, j)] = r

            for j in range(i + 1, len(last_pwords)):
                r = rx[(i, j)]
                if r:
                    i_rhyming = j - i
                    break

            rhyme_graph.append(i_rhyming)
        rhyme_graph.append(None)  # последняя строка

        nlines = len(last_pwords)

        # TODO: переделать на формирование строки rhyme_scheme так, как это сделано в EnglishPoetryScansionTool

        rhyme_scheme = '-' * len(last_pwords)

        rhyme_graph_str = ' '.join(map(str, [(0 if e is None else e) for e in rhyme_graph]))
        if nlines == 4:
            if rhyme_graph_str == '2 0 1 0':
                rhyme_scheme = 'A-AA'
            else:
                r01 = rx[0, 1]
                r02 = rx[0, 2]
                r03 = rx[0, 3]
                r12 = rx[1, 2]
                r13 = rx[1, 3]
                r23 = rx[2, 3]

                if r01 and r12 and r23:
                    # 22.04.2022 отдельно детектируем рифмовку AAAA, так как она зачастую выглядит очень неудачно и ее
                    # желательно устранять из обучающего датасета.
                    rhyme_scheme = 'AAAA'
                elif r02 and r13:
                    rhyme_scheme = 'ABAB'
                elif r03 and r12:
                    rhyme_scheme = 'ABBA'
                # 22-12-2021 добавлена рифмовка AABB
                elif r01 and r23:
                    rhyme_scheme = 'AABB'
                # 28-12-2021 добавлена рифмовка "рубаи" AABA
                elif r01 and r03 and not r02:
                    rhyme_scheme = 'AABA'
                elif r01 and r02 and r12 and not r13 and not r13:
                    rhyme_scheme = 'AAAB'
                # 21.05.2022 проверяем неполные рифмовки A-A- и -A-A
                elif r02 and not r13:
                    rhyme_scheme = 'A-A-'
                elif not r02 and r13:
                    rhyme_scheme = '-A-A'
                elif r12 and not r01 and not r23:
                    rhyme_scheme = '-AA-'
                elif r03 and not r12:
                    rhyme_scheme = 'A--A'
                elif not r01 and r23:
                    rhyme_scheme = '--AA'
                elif r01 and not r23:
                    rhyme_scheme = 'AA--'
                #else:
                #    rhyme_scheme = '----'
        elif nlines == 5:
            if rhyme_graph_str == '1 3 1 0 0':
                # Говоря́т что Толста́я Татья́на
                # В мужика́х всё иска́ла изъя́ны.
                # Как оты́щет хоть что́-то,
                # Посыла́ет в два счё́та.
                # И они́ все иду́т, как бара́ны.
                rhyme_scheme = 'AABBA'
            elif rhyme_graph_str == '1 1 0 1 0':
                # Мужчина тот червонной масти -
                # В его мое сердечко власти
                # Поет об ураганной страсти,
                # О том, что в мире этом нет
                # Любви одной на много лет.
                rhyme_scheme = 'AAABB'
            elif rhyme_graph_str == '3 1 2 0 0':
                # Вы слуги любви, но какие неверные слуги.
                # Волнующи речи. Сладки дорогие духи.
                # Вы пылки и страстны. Когда ж отпоют петухи,
                # Вы вновь возвратитесь к постели любимой подруги,
                # Сцеловывать с губ ее пламенных те же грехи.
                rhyme_scheme = 'ABBAB'
            elif rhyme_graph_str == '0 1 2 0 0':
                # Цепями нужд обременённый,
                # Без друга, в горе и слезах
                # Погиб ты... При чужих водах
                # Лежит, безгласный и забвенный,
                # Многострадальческий твой прах.
                rhyme_scheme = '-AA-A'
            elif rhyme_graph_str == '1 0 1 0 0':
                # Мой кора́блик плыве́т по волна́м
                # И курси́рует зде́сь он и та́м
                # Я куда́ захочу́
                # Его поворочу́
                # Ведь кора́блик - бума́жный обма́н
                rhyme_scheme = 'AABB-'
            elif rhyme_graph_str == '2 3 1 0 0':
                # И стоит он, околдован,
                # Не мертвец и не живой -
                # Сном волшебным очарован,
                # Весь опутан, весь окован
                # Лёгкой цепью пуховой.
                rhyme_scheme = 'ABAAB'
            elif rhyme_graph_str == '1 3 0 0 0':
                # Как-то зна́л я деви́цу - спортсме́нку,
                # Что ногтя́ми цепля́лась за сте́нку,
                # Я проси́л: " прекрати́,
                # Мне обо́и не рви́! "
                # Ноль в отве́т, как горо́хом об сте́нку.
                rhyme_scheme = 'AA--A'
            elif rhyme_graph_str == '2 0 1 0 0':
                # Спя́т поля́ под покрыва́лом бе́лым,
                # Спя́т дере́вья в сне́жном серебре́.
                # Всё́ зима́ забо́тливо оде́ла,
                # Да́же ре́чку льдо́м укры́ть успе́ла,
                # И моро́з гуля́ет по земле́.
                rhyme_scheme = 'A-AA-'
            elif rhyme_graph_str == '4 0 1 0 0':
                # Мужичо́нка плюга́вый из За́мбии
                # До печё́нок был спи́ртом отра́влен, и
                # Только бра́гу лака́л,
                # Что из пчё́л выжима́л,
                # Обрыва́я им кры́лышки в За́мбии.
                rhyme_scheme = 'A-BBA'
            elif rhyme_graph_str == '1 1 1 1 0':
                # Соли́дной стару́шке в меха́х
                # Броса́ла серё́жки ольха́.
                # Ах, ка́к я плоха́!
                # Смея́лась ольха́
                # И па́чкала да́ме меха́.
                rhyme_scheme = 'AAAAA'
            elif rhyme_graph_str == '0 0 1 0 0':
                # На берегу́, где бу́хты си́нь,
                # И го́ры - полукру́жьем,
                # И кра́н торчи́т, - вдали́, оди́н...
                # Сиде́л он, но́ги из штани́н
                # Слегка́ откры́в нару́жу.
                rhyme_scheme = '--AA-'
            elif rhyme_graph_str == '0 3 1 0 0':
                # А мы́шцы! Пле́чи! Мо́щный то́рс!
                # И без жири́нки - ля́жки!
                # Волни́стый ря́д густы́х воло́с!
                # И гре́ческий, как бу́дто, но́с.
                # И ше́рсть из-под руба́шки.
                rhyme_scheme = '-ABBA'
            elif rhyme_graph_str == '0 0 2 0 0':
                # Сло́во сле́ва,
                # Сло́во спра́ва,
                # Ме́жду ни́ми я́ - пробе́л,
                # Незави́симая я́ма
                # Пустоте́лость не у де́л.
                rhyme_scheme = '--A-A'
            elif rhyme_graph_str == '4 1 1 0 0':
                # Цвету́т весе́нние сады́
                # Ажу́рным о́блаком пред взо́ром.
                # Проши́т иску́снейшим узо́ром
                # Убру́с печа́ли, под кото́рым,
                # Как под шатро́м, скрыва́лась ты́.
                rhyme_scheme = 'ABBBA'
            elif rhyme_graph_str == '2 2 2 0 0':
                # Что ме́жду ма́ртом и зимо́й?
                # Заче́м мете́ли всю́ неде́лю,
                # Заче́м колю́чею и зло́й
                # Закостене́лые капе́ли
                # Стоя́т под кры́шею стено́й?
                rhyme_scheme = 'ABABA'
            elif rhyme_graph_str == '4 0 0 0 0':
                # Заче́м блести́т снег, ка́к ого́нь
                # Упа́вших звё́зд в моме́нт были́нный
                # И стё́ртый межсезо́нной мгло́й,
                # Где миллио́нная снежи́нка
                # Уже́ лети́т к тебе́ в ладо́нь?
                rhyme_scheme = 'A---A'
            elif rhyme_graph_str == '3 3 0 0 0':
                # От рома́шек махро́вых светло́,
                # Дивный ве́чер объя́т тишино́ю.
                # И как ра́ньше поко́рно, легко́
                # Внятно слы́шу её́ же крыло́
                # Над свое́й я смире́нной душо́ю.
                rhyme_scheme = 'AB-AB'
            elif rhyme_graph_str == '2 2 0 1 0':
                # Вы́купай меня́ в свое́й любви́,
                # Оплети́ рома́шкой и шалфе́ем,
                # На блины́ с варе́ньем позови́,
                # До дождя́ прийти́ к тебе́ успе́ю,
                # И косну́ться пря́ных гу́б посме́ю,
                rhyme_scheme = 'ABABB'
            elif rhyme_graph_str == '2 2 0 0 0':
                # Поэ́зия должна́ быть страннова́та,
                # Поэ́зия совсе́м не два́жды два́.
                # Для одного́ она́, как стеклова́та,
                # А для друго́го про́сто - тры́н трава́,
                # А во́т для тре́тьего, как два́ крыла́.
                rhyme_scheme = 'ABAB-'
            elif rhyme_graph_str == '0 1 0 1 0':
                # Вы — поводырь, а я — слепой старик.
                # Вы — проводник. Я еду без билета!
                # Иной вопрос остался без ответа,
                # И втоптан в землю прах друзей моих.
                # Вы — глас людской. Я — позабытый стих.
                rhyme_scheme = '-AABB'
            elif rhyme_graph_str == '1 0 0 0 0':
                # Одна странная дама с Камчатки
                # Одевала на ноги перчатки
                # говорила я утка
                # и задрав к верху юбку
                # начинала носиться по грядкам
                rhyme_scheme = 'AA---'
            else:
                rhyme_scheme = '-----'
        elif nlines == 6:
            rhyme_scheme = convert_rhyme_graph_to_scheme(rhyme_graph)

            # if rhyme_graph_str == '1 3 1 0 1 0':
            #     # То воплощаюсь морскою звездой, то притворяюсь ознобом тумана.
            #     # Изобличаемый их простотой, пробую мудрую жизнь океана.
            #     # Солоно-горько слезится вода неимоверны людские глубины.
            #     # На полюсах замерзают года в две исполинские льдины.
            #     # Холод глубин порождает разрыв - айсберг красуется сгустком тумана.
            #     # Солнцем волшебные ткутся ковры - славный презент экзотическим странам.
            #     rhyme_scheme = 'AABBAA'
            # elif rhyme_graph_str == '1 1 1 0 1 0':
            #     # Полтинник, это много или мало?
            #     # Прожить б ещё полтинник не мешало.
            #     # Ещё разок хочу начать сначала,
            #     # Но жизнь идёт, и остаётся мало...
            #     # Такая уж судьба у человека,
            #     # Ведь мало кто живёт длиннее века.
            #     rhyme_scheme = 'AAAABB'
            # elif rhyme_graph_str == '2 2 2 2 0 0':
            #     # Что мечты мои волнует
            #     # На привычном ложе сна?
            #     # На лицо и грудь мне дует
            #     # Свежим воздухом весна,
            #     # Тихо очи мне целует
            #     # Полуночная луна.
            #     rhyme_scheme = 'ABABAB'
            # elif rhyme_graph_str == '1 0 3 1 0 0':
            #     # Ка́к цари́це е́й служи́ли,
            #     # Сто́лько сре́дств в неё́ вложи́ли,
            #     # А отда́чи - никако́й!
            #     # Та́к и вы́гнали из кле́тки -
            #     # Пу́сть идё́т и жрё́т объе́дки
            #     # У столо́вки заводско́й.
            #     rhyme_scheme = 'AABCCB'
            # elif rhyme_graph_str == '1 0 1 0 1 0':
            #     # Очки́ смея́лись над глаза́ми:
            #     # - Мы пе́рвые, а вы́ за на́ми!
            #     # Кичи́лся делово́й костю́м:
            #     # - Собо́й я затмева́ю у́м!
            #     # Боти́нки гро́мче все́х крича́ли:
            #     # - На на́с все де́ржится, слыха́ли?
            #     rhyme_scheme = 'AABBCC'
            # elif rhyme_graph_str == '2 2 0 0 1 0':
            #     # И я́ почти́ Ома́р Хайя́м
            #     # Во-пе́рвых мы́ одно́й конфе́ссии
            #     # Он бы́л врачо́м когда́-то та́м
            #     # И я́, счита́й, двойно́й профе́ссии
            #     # Он бы́л врачо́м ещё́ поэ́том
            #     # А я́ поэ́т, шофё́р при э́том
            #     rhyme_scheme = 'ABABCC'
            # elif rhyme_graph_str == '1 2 3 1 0 0':
            #     # Но гло́жет мо́зг сомне́ние,
            #     # Како́е-то́ - волне́ние.
            #     # Неи́стово шуми́т в бачке́ вода́!
            #     # К утру́ оцепене́ние,
            #     # И я́ друго́го мне́ния -
            #     # Пора́ бы мне́ наве́даться туда́.
            #     rhyme_scheme = 'AABCCB'
            # elif rhyme_graph_str == '1 1 0 1 1 0':
            #     # Возле топкого болота
            #     # На большого бегемота
            #     # Накатила вдруг зевота:
            #     # Назеваться хочет всласть,
            #     # Распахнул пошире пасть.
            #     # Не спеши в нее попасть.
            #     rhyme_scheme = 'AAABBB'
            # else:
            #     rhyme_scheme = '------'
        elif nlines == 3:
            if rhyme_graph_str == '0 0 0':
                rhyme_scheme = '---'

            elif rhyme_graph_str == '1 0 0':
                # А людям пример их — наука,
                # Что двигаться лишняя мука,
                # Что горшее зло — суета
                rhyme_scheme = 'AA-'

            elif rhyme_graph_str == '2 0 0':
                rhyme_scheme = 'A-A'

            elif rhyme_graph_str == '0 1 0':
                # Нахально блещущих, развратных и мятежных,
                # Так мускус, фимиам, пачули и бензой
                # Поют экстазы чувств и добрых сил прибой.
                rhyme_scheme = '-AA'

            elif rhyme_graph_str == '1 1 0':
                rhyme_scheme = 'AAA'
        elif nlines == 2:
            if rhyme_graph_str == '1 0':
                rhyme_scheme = 'AA'

        edges = [(0 if e is None else e) for e in rhyme_graph]
        for i, e in enumerate(edges):
            if e>0 and edges[i + e]==0:
                edges[i + e] = -e

        num_rhymed_lines = sum(e!=0 for e in edges)
        if nlines == 3:
            # Для 3-строчников требуем только 1 пару рифмованных строк.
            if num_rhymed_lines < 2:
                rhyme_score = 1 - 1.0/6.0
            else:
                rhyme_score = 1.0
        else:
            # за каждую нерифмованную строку даем такой штраф:
            rhyme_penalty1 = 1.0 / (2*nlines)

            # подсчитываем штрафы за нерифмованные строки
            rhyme_penalty = sum((rhyme_penalty1 if e==0 else 0.0) for e in edges)

            rhyme_score = 1.0 - rhyme_penalty

        return rhyme_scheme, rhyme_score, rhyme_graph

    def get_dolnik_patterns(self, num_syllables: List[int]):
        # Подготовим варианты идеальной расстановки ударений для дольников
        dolnik_patterns = []

        max_num_syllables = max(num_syllables)
        min_num_syllables = min(num_syllables)

        if max_num_syllables == 21:
            # То воплощаюсь морскою звездой, то притворяюсь ознобом тумана.
            # Изобличаемый их простотой, пробую мудрую жизнь океана.
            # Солоно-горько слезится вода неимоверны людские глубины.
            # На полюсах замерзают года в две исполинские льдины.
            # Холод глубин порождает разрыв - айсберг красуется сгустком тумана.
            # Солнцем волшебные ткутся ковры - славный презент экзотическим странам.
            dolnik_patterns.append([[1,0,0,1,0,0,1,0,0,1, 1,0,0,1,0,0,1,0,0,1, 0]])


        if min_num_syllables == 15 and max_num_syllables == 15:
            # Белого снега вращенье в блеске ночных фонарей...
            # Мудрого мира ученье я позабуду скорей.
            # Помнить не стану о мире, где не находят любовь,
            # В сумраке старой квартиры стынет от холода кровь.
            # 100010101001001
            dolnik_patterns.append([[1,0,0, 1,0,0, 1,0,1, 0,0,1, 0,0,1]])

        if min_num_syllables == 13 and max_num_syllables == 13:
            # Я мечтою ловил уходящие тени,
            # Уходящие тени погасавшего дня,
            # Я на башню всходил, и дрожали ступени,
            # И дрожали ступени под ногой у меня.
            dolnik_patterns.append([[0,0,1,0,0,1,0,0,1,0,0,1,0],
                                    [0,0,1,0,0,1,0,1,0,1,0,0,1]])

        if min_num_syllables >= 12 and max_num_syllables == 13:
            # 27.03.2025
            # Я так мечтал о море, о южных берегах
            # И с детства был им болен, его я видел в снах
            # Однажды сел я в поезд, и вот мечта сбылась
            # Передо мною море, шум волн и далей гладь.
            dolnik_patterns.append([[0,1,0,1,0,1,0,0,1,0,1,0,1]])


        if min_num_syllables == 13 and max_num_syllables == 14:
            # Моргает первый. Во как! Вот это интересно!
            # Но всё ж не понимаю! Иль я такой дебил?
            # Бурчит второй. Тебе я могу сознаться честно.
            # Половником вот этим всю морду бы разбил!
            dolnik_patterns.append([[0,1,0,1,0,1, 0, 0,1,0,1,0,1,0,],
                                    [0,1,0,1,0,1, 0, 0,1,0,1,0,1,]])

        if min_num_syllables == 14 and max_num_syllables == 14:
            # И тяжело на сердце: что с грузом делать этим?
            # Лучину в сердце надо, совсем немного света,
            # Чтобы комок сомнений из грустных размышлений
            # Растаял будто льдинки, исчезнув легкой тенью.
            dolnik_patterns.append([[0,1,0,1,0,1,0, 0,1,0,1,0,1,0]])

        if max_num_syllables == 18 and min_num_syllables == 18:
            dolnik_patterns.append([[0,1,0,1,0,1,0,1,0, 0,1,0,1,0,1,0,1,0]])

        if min_num_syllables == 22 and max_num_syllables == 22:
            dolnik_patterns.append([[0,1,0,1,0,1,0,1,0,1,0, 0,1,0,1,0,1,0,1,0,1,0]])

        if min_num_syllables == 17 and max_num_syllables == 17:
            # Неумолимо время тает в объятьях горестной тоски...
            # О как же Вас мне не хватает! Как не хватает мне любви!
            # Вас позабыть - не в моих силах, Вас невозможно позабыть!
            # Вы близко, но увы, с другою... Но Вас сумела я простить.
            dolnik_patterns.append([[0,1,0,1,0,1,0,1, 0, 0,1,0,1,0,1,0,1]])

        if len(num_syllables) == 4 and num_syllables == [9, 10, 10, 9]:
            # Без тебя тяжело и пусто,
            # Как в забытом, заброшенном доме...
            # Ни о чём я не думаю, кроме
            # Как о прошлом, тихо и грустно.
            #
            # 001001010
            # 0010010010
            # 0010010010
            # 001001010
            dolnik_patterns.append([[0,0,1,0,0,1,0,1,0],
                                    [0,0,1,0,0,1,0,0,1,0],
                                    [0,0,1,0,0,1,0,0,1,0],
                                    [0,0,1,0,0,1,0,1,0]
                                    ])

        if max_num_syllables == 6:
            # Солнышко с потолка
            # Кинет лучи на стол...
            # Чудо на три рожка -
            # Триста уже не сто.
            dolnik_patterns.append([[1,0,0,1,0,1]])

        if min_num_syllables == 6 and max_num_syllables == 7:
            # В городе ночь без края,
            # Что волшебства полна.
            # Света, души Весна
            # Дышит, Любовь спасая.
            dolnik_patterns.append([[1,0,0,1,0,1,0],
                                    [1,0,0,1,0,1],
                                    [1,0,0,1,0,1],
                                    [1,0,0,1,0,1,0]])

        if min_num_syllables == 7 and max_num_syllables == 7:
            # Божью Любовь Святую
            # Миру откроем, братья.
            # Тех, кто идёт вслепую,
            # Примем в свои объятья.
            dolnik_patterns.append([[1,0,0,1,0,1,0],
                                    [0,1,0,1,0,1,0],
                                    [0,1,0,1,0,1,0],
                                    [1,0,0,1,0,1,0]])

            # Вы спросите - третий кто?
            # А я улыбнусь в ответ.
            # Он будет стоять среди
            # И скалиться, будто черт.
            dolnik_patterns.append([[0,1,0,0,1,0,0]])

        if min_num_syllables == 7 and max_num_syllables == 8:
            # На белом коне ты скачешь,
            # В ноздрях его пыл огня,
            # Копьём ты врага пронзаешь,
            # В груди его месть твоя.
            dolnik_patterns.append([[0,1,0,0,1,0,1,0],
                                    [0,1,0,0,1,0,1]])

            # Ангелы с нами. С нами.
            # Они и хранят, и лечат,
            # Бывает, не спят ночами.
            # Держи мою руку крепче.
            dolnik_patterns.append([[1,0,0,1,0,1,0,0],
                                    [0,1,0,0,1,0,1,0],
                                    [0,1,0,0,1,0,1,0],
                                    [0,1,0,0,1,0,1,0]])


        if max_num_syllables == 8:
            dolnik_patterns.append([[1, 0, 0, 1, 1, 0, 0, 1]])

        if min_num_syllables == 8 and max_num_syllables == 8:
            # Лег над про́пастью ру́сский путь.
            # И срыва́ется в бе́здну даль.
            # Русский ру́сского не забудь.
            # Русский ру́сского не предай.
            dolnik_patterns.append([[1,0,1,0,0,1,0,1]])


        if min_num_syllables == 8 and max_num_syllables == 9:
            # И теперь я не знаю боли,
            # Все недуги мои прошли.
            # И теперь я стихи слагаю
            # О великой твоей любви
            dolnik_patterns.append([[0,0,1,0,0,1,0,1,0],
                                    [0,0,1,0,0,1,0,1]])

        if min_num_syllables == 8 and max_num_syllables == 10:
            # Конца не вижу я испытанью!
            # Мешок тяжел, битком набит!
            # Куда деваться мне с этой дрянью?
            # Хотела выпустить — сидит.
            dolnik_patterns.append([[0,1,0,1,0,1,0,0,1,0],
                                    [0,1,0,1,0,1,0,1]])

            # Как трехсотая, с передачею,
            # Под Крестами будешь стоять
            # И своей слезою горячею
            # Новогодний лед прожигать.
            dolnik_patterns.append([[1,0,1,0,1,0,0,1,0,0],
                                    [1,0,1,0,1,0,0,1]])


        if min_num_syllables == 9 and max_num_syllables == 9:
            # Бог, который весь мир расчислил,
            # Угадал ее злые мысли
            # И обрек ее на несчастье,
            # Разорвал ее на две части.
            dolnik_patterns.append([[0, 0, 1, 0, 0, 1, 0, 1, 0],
                                    [0, 0, 1, 0, 0, 1, 0, 1, 0],
                                    [0, 0, 1, 0, 1, 0, 0, 1, 0],
                                    [0, 0, 1, 0, 1, 0, 0, 1, 0],
                                    ])

        if max_num_syllables == 9:
            # Меня уносит куда-то вниз.
            # Хотя мне кажется, что наверх.
            # И кто-то лучший выходит из
            # И снова прячется ото всех.
            dolnik_patterns.append([[0, 1, 0, 1, 0, 0, 1, 0, 1]])

            # Голос яшмовой флейты слышу.
            # Кто же это играет на ней?
            # Ночь. Беседка. Луна все выше,
            # На душе все грустней и грустней.
            dolnik_patterns.append([[0,0,1,0,0,1,0,1,0], [0,0,1,0,0,1,0,0,1]])

        if min_num_syllables == 9 and max_num_syllables == 10:
            # А в доме эхо уронит чашку,
            # ложное эхо предложит чай,
            # ложное эхо оставит на ночь,
            # когда ей надо бы закричать:
            dolnik_patterns.append([[0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
                                    [1, 0, 0, 1, 0, 0, 1, 0, 1],
                                    [1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
                                    [0, 1, 0, 1, 0, 0, 1, 0, 1]
                                    ])

        if min_num_syllables == 10 and max_num_syllables == 10:
            dolnik_patterns.append([[1, 0, 0, 0, 1, 1, 0, 0, 0, 1]])

            # Чтоб придти туда, где жизнь вечная:
            # В океан любви светлой Вечности,
            # Где сегодня ты пребываешь. Жди,
            # Я приду к тебе, чтоб нам вместе быть.
            dolnik_patterns.append([[1, 0, 1, 0, 1,  1, 0, 1, 0, 1]])

            # Мотивом лучших стихов и песен
            # От сердца к сердцу: Христос Воскресе!
            # Нет лишних в этой бессмертной пьесе.
            # Святая Пасха: Христос Воскресе!
            dolnik_patterns.append([[0,1,0,1,0, 0,1,0,1,0]])

            # Ночь заправляет крутыми днями.
            # Площадь отравлена голубями.
            # Хитрым зигзагом бежит прямая...
            # Все поголовно всё понимают.
            dolnik_patterns.append([[1,0,0,1,0,0,1, 0,1,0]])

            # Не жаль деревянный мой карандаш.
            # Он самый желанный. Он - твой. Он - наш!
            # Не жаль деревянный мой карандаш.
            # И холст белотканный. И мой пейзаж.
            dolnik_patterns.append([[0,1,0,0,1,0,0,1, 0,1]])

        if min_num_syllables == 10 and max_num_syllables == 11:
            # Мимо бе́лого я́блока луны́,
            # Мимо кра́сного я́блока зака́та
            # Облака́ из неве́домой стра́ны
            # К нам спеша́т, и опя́ть бегут куда́-то.
            dolnik_patterns.append([[0,0,1,0,0,1,0,0,0,1], [0,0,1,0,0,1,0,1,0,1,0]])

            # Поглотила рутина бумажная,
            # Убивая время для творчества.
            # Вдохновения муза миражная
            # От обиды мается, корчится.
            dolnik_patterns.append([[0,0,1,0,0,1,0,0,1,0,0],
                                    [0,0,1,0,1,0,0,1,0,0]])

        if max_num_syllables == 12:
            # Ров противотанковый вырыт у реки
            # Миром всем копали - дети, старики.
            # От фашистов Жуковку с Запада спасал,
            # Комсомол помощников из Москвы прислал.
            # Так Десны излучину ров соединил,
            # Люди здесь работали из последних сил.
            # Как напоминание ров нам о войне,
            # Сила духа Русского видится в том рве!
            dolnik_patterns.append([[1,0,1,0,1,0, 0,1,0,1,0,1]])

        if min_num_syllables == 11 and max_num_syllables == 11:
            # Сердце скорее рвется, чтобы дойти.
            # Много ли остается дней впереди?
            # Скоро ль увижу небо, встречу Христа,
            # Кто мне дороже хлеба в жизни сей стал?
            dolnik_patterns.append([[1,0,0,1,0,1,0,1,0,0,1]])

        if min_num_syllables == 11 and max_num_syllables == 12:
            # Мрачный полёт, долго лечу я с трамплина.
            # В лыжи я врос. Не контролирую лыж.
            # Кровь холодна. Что же, душа, не бурлишь?
            # В небе веду тяжкий с собой поединок.
            dolnik_patterns.append([[1,0,0,1,1,0,0,1,0,0,1,0],
                                    [1,0,0,1,1,0,0,1,0,0,1]])

            # Переле́тным ра́дуюсь пти́цам от души́
            # С ни́ми прилета́ют по весне́ коты́
            dolnik_patterns.append([[1,0,1,0,1,0,1, 1,0,1,0,1],
                                    [1,0,1,0,1,0,1,  0,1,0,1]])


        if min_num_syllables == 11 and max_num_syllables == 13:
            # Вокруг сгустятся тучи, зашлёпают в ночи
            # Под пиччикато проклятые капли.
            # Ты, молнии подобный, смычок загрохочи,
            # Чтоб на ковчеге схватки не ослабли.
            dolnik_patterns.append([[0,1,0,1,0,1, 0, 0,1,0,1,0,1],
                                    [0,1,0,1,0,1,0,1,0,1,0]])

        if min_num_syllables == 13 and max_num_syllables == 13:
            # Научи меня терпеть, научи меняться.
            # Даже если не дышу, научи смеяться.
            # Научи вставать опять, если нету силы.
            # Научи меня прощать все мои обиды.
            dolnik_patterns.append([[1,0,1,0,1,0,1,0, 0,1,0,1,0,]])

        if max_num_syllables in (13, 14):
            # Вырастет из сына свин, если сын — свинёнок
            dolnik_patterns.append([[1,0,1,0,1,0,1, 1,0,1,0,1,0,1]])

        if max_num_syllables == 14:
            dolnik_patterns.append([[1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1]])

        if min_num_syllables == 15 and max_num_syllables == 15:
            # В день рожденья моего, а точнее - этой ночью,
            # Что-то вдруг произошло, отверзающее очи.
            # Это "что-то" я теперь в сердце сохраню навечно:
            # Мне была открыта Дверь, за которой - Бесконечность.
            dolnik_patterns.append([[1,0,1,0,1,0,1, 1,0,1,0,1,0,1,0,]])

        if min_num_syllables == 15 and max_num_syllables == 16:
            # Жду мелодию строки, как всевышнее послание,
            # Снова в зеркале реки отразилось солнце раннее.
            # Обертоны певчих птиц-переливы перламутра,
            # И взволнованность ресниц, распахнувших это утро.
            dolnik_patterns.append([[1,0,1,0,1,0,1, 1,0,1,0,1,0,1,0,1]])

        if max_num_syllables == 16:
            # Распустила косу русую, - проскользнула в рожь коса
            # И скосила острым волосом звездоликий василёк.
            # Улыбнулась, лепестковая, и завился мотылёк -
            # Не улыбка ль воплощённая?... Загудело, как оса...
            dolnik_patterns.append([[1,0,1,0,1,0,1,0, 0,1,0,1,0,1,0,1]])


        if 7 <= max_num_syllables <= 8:
            dolnik_patterns.append([[0,1,0,1,0,0,1,0][:max_num_syllables]])

        if 8 <= max_num_syllables < 9:
            dolnik_patterns.append([[0,0,1,0,0,1,0,1,0][:max_num_syllables]])

        if 10 <= max_num_syllables <= 12:
            dolnik_patterns.append([[0,0,1,0,0,1,0,0,0,1,0,0][:max_num_syllables]])

        if 13 <= max_num_syllables <= 15:
            dolnik_patterns.append([[0,0,1,0,0,1,0,0,1,0,0,0,1,0,0][:max_num_syllables]])

        if min_num_syllables == 17 and max_num_syllables == 17:
            # Но если ты заявишь гордо: "Моя краса живет в веках!" -
            # Улыбка, взгляд - как два аккорда у малыша и старика -
            # Наследник юный, подрастая, продолжит дел твоих отсчет
            # И подтвердит, и оправдает, и принесет тебе почет.
            dolnik_patterns.append([[0,1,0,1,0,1,0,1, 0, 0,1,0,1,0,1,0,1]])


        if min_num_syllables == 19 and max_num_syllables == 20:
            # Жизнь прекрасна своим бесподобием, вот и я не похож на Христа,
            # Не смотри на меня исподлобия, я к тебе подошёл неспроста!
            # Подошёл, словно, ёжик в тумане, как петух, сосчитавший всех кур,
            # Как альфонс в авантюрном романе, как добытчик пушнины и шкур!
            dolnik_patterns.append([[0,0,1,0,0,1,0,0,1,0,0,1, 0, 1,0,0,1,0,0,1],
                                    [0,0,1,0,0,1,0,0,1,0,0,1, 0, 1,0,0,1,0,0,1],
                                    [0,0,1,0,0,1,0,0,1, 0, 0,0,1,0,0,1,0,0,1],
                                    [0,0,1,0,0,1,0,0,1, 0, 0,0,1,0,0,1,0,0,1]])


        return dolnik_patterns

    def align_artishok(self, lines, plines):
        stressed_words_groups = [StressVariantsSlot.build(poetry_words=pline.pwords, aligner=self, allow_stress_shift=True, allow_unstress12=True) for pline in plines]

        # Первые три строки - амфибрахий, последняя - хорей.
        line_meters = [('амфибрахий', (0, 1, 0))]*3 + [('хорей', (1, 0))]

        best_scores = dict()

        # В каждой строке перебираем варианты расстановки ударений.
        for ipline, (pline, (metre_name, metre_signature)) in enumerate(zip(plines, line_meters)):
            best_scores[ipline] = dict()

            cursor = MetreMappingCursor(metre_signature, prefix=0)
            for metre_mapping in cursor.map(stressed_words_groups[ipline], self):
                stressed_words = [m.word for m in metre_mapping.word_mappings]
                new_stress_line = LineStressVariant(pline, stressed_words, self)

                if new_stress_line.get_rhyming_tail().is_ok():
                    tail_str = new_stress_line.get_rhyming_tail().__repr__()
                    score = metre_mapping.get_score()
                    if tail_str not in best_scores[ipline]:
                        prev_score = -1e3
                    else:
                        prev_score = best_scores[ipline][tail_str][0].get_score()
                    if score > prev_score:
                        best_scores[ipline][tail_str] = (metre_mapping, new_stress_line)

        # Теперь для каждой исходной строки имеем несколько вариантов расстановки ударений.
        # Перебираем сочетания этих вариантов, проверяем рифмовку и оставляем лучший вариант для данного метра.
        stressed_lines2 = [list() for _ in range(len(best_scores))]
        for iline, items2 in best_scores.items():
            stressed_lines2[iline].extend(items2.values())

        best_score = 0.0
        best_metre = None
        best_rhyme_scheme = None
        best_variant = None
        best_rhyme_graph = None

        rhyming_detection_cache = dict()

        vvx = list(itertools.product(*stressed_lines2))
        for ivar, plinev in enumerate(vvx):
            # plinev это набор из двух экземпляров кортежей (MetreMappingResult, LineStressVariant).

            # Определяем рифмуемость
            rhyme_scheme = None
            last_pwords = [pline[1].get_rhyming_tail() for pline in plinev]

            rhyme_scheme, rhyme_score, rhyme_graph = self.detect_rhyming(last_pwords, rhyming_detection_cache)

            total_score = rhyme_score * mul([pline[0].get_score() for pline in plinev])
            if total_score > best_score:
                best_score = total_score
                best_metre = 'амфибрахий'
                best_rhyme_scheme = rhyme_scheme
                best_variant = plinev
                best_rhyme_graph = rhyme_graph

        if best_variant is None:
            # В этом случае вернем результат с нулевым скором и особым текстом, чтобы
            # можно было вывести в лог строки с каким-то дефолтными
            return PoetryAlignment.build_no_rhyming_result([pline.get_first_stress_variants(self) for pline in plines])
        else:
            # Возвращаем найденный вариант разметки и его оценку
            best_lines = [v[1] for v in best_variant]
            metre_mappings = [v[0] for v in best_variant]
            return PoetryAlignment(best_lines, best_score, best_metre,
                                   rhyme_scheme=best_rhyme_scheme,
                                   rhyme_graph=best_rhyme_graph,
                                   metre_mappings=metre_mappings)

    def align4(self, lines, check_rhymes):
        for line in lines:
            tokens = tokenize(line)
            ntokens = len(tokens)
            if ntokens > self.max_words_per_line:
                raise ValueError('Line is too long @2008: max_words_per_line={} tokens={} line="{}"'.format(self.max_words_per_line, '|'.join(tokens), line))

        plines = [PoetryLine.build(line, self.udpipe, self.accentuator) for line in lines]

        # 08.11.2022 добавлена защита от взрыва числа переборов для очень плохих генераций.
        for pline in plines:
            if sum((pword.n_vowels>=1) for pword in pline.pwords) >= self.max_words_per_line:
                raise ValueError('Line is too long @1983: "{}"'.format(pline))

        n_sylla = tuple(l.get_num_syllables() for l in plines)
        if all((line == line.lower()) for line in lines):
            if n_sylla == (11, 9, 11, 2):
                # Попробуем разметить как артишок
                res = self.align_artishok(lines, plines)
                return res

        best_score = 0.0
        best_metre = None
        best_rhyme_scheme = None
        best_variant = None
        best_rhyme_graph = None
        rhyming_detection_cache = dict()

        # Проверяем основные метры.
        for allow_stress_shift in [True]:  # [False, True]:
            if best_score > self.early_stopping_threshold_score:
                break

            # Заранее сгенерируем для каждого слова варианты спеллчека и ударения...
            stressed_words_groups = [StressVariantsSlot.build(poetry_words=pline.pwords, aligner=self, allow_stress_shift=allow_stress_shift, allow_unstress12=True) for pline in plines]

            check_meters = meters
            if n_sylla == (9, 8, 9, 2):
                # Порошки - только ямб
                check_meters = [('ямб', (0, 1))]

            # Для каждой строки перебираем варианты разметки и оставляем по ~2 варианта в каждом метре.
            for metre_name, metre_signature in check_meters:
                best_scores = dict()

                # В каждой строке перебираем варианты расстановки ударений.
                for ipline, pline in enumerate(plines):
                    best_scores[ipline] = dict()

                    for prefix in self.get_prefixes_for_meter(metre_signature):
                        cursor = MetreMappingCursor(metre_signature, prefix=prefix)
                        for metre_mapping in cursor.map(stressed_words_groups[ipline], self):
                            stressed_words = [m.word for m in metre_mapping.word_mappings]
                            new_stress_line = LineStressVariant(pline, stressed_words, self)

                            if new_stress_line.get_rhyming_tail().is_ok():
                                tail_str = new_stress_line.get_rhyming_tail().__repr__()
                                score = metre_mapping.get_score()
                                if tail_str not in best_scores[ipline]:
                                    prev_score = -1e3
                                else:
                                    prev_score = best_scores[ipline][tail_str][0].get_score()
                                if score > prev_score:
                                    best_scores[ipline][tail_str] = (metre_mapping, new_stress_line)

                # Теперь для каждой исходной строки имеем несколько вариантов расстановки ударений.
                # Перебираем сочетания этих вариантов, проверяем рифмовку и оставляем лучший вариант для данной метра.
                stressed_lines2 = [list() for _ in range(len(best_scores))]
                for iline, items2 in best_scores.items():
                    stressed_lines2[iline].extend(items2.values())

                vvx = list(itertools.product(*stressed_lines2))
                for ivar, plinev in enumerate(vvx):
                    # plinev это набор из двух экземпляров кортежей (MetreMappingResult, LineStressVariant).

                    # Различные дефекты ритма
                    metre_defects_score = 1.0
                    # 11.12.2022 сдвиг одной строки на 1 позицию
                    nprefixa = collections.Counter(pline[0].prefix for pline in plinev)
                    if nprefixa.get(0) == 1 or nprefixa.get(1) == 1:
                        metre_defects_score *= 0.1

                    # Есть жанры, где в четырех строках должно быть 3 варианта слоговой длины.
                    # Примеры: порошки (9-8-9-2) и артишоки (11-9-11-2).
                    # Для них не штрафуем за 3 варианта длины!
                    n_sylla = tuple(l[1].poetry_line.get_num_syllables() for l in plinev)
                    if n_sylla not in [(11, 9, 11, 2), (9, 8, 9, 2)]:
                        nsyllaba = collections.Counter(len(pline[1].stress_signature) for pline in plinev)
                        if len(nsyllaba) > 2:
                            # Есть более 2 длин строк в слогах
                            metre_defects_score *= 0.1
                        else:
                            for nsyllab, num in nsyllaba.most_common():
                                if num == 3:
                                    # 03.05.2024 особый случай с рубаи. третья строка обычно отличается по числу слогов.
                                    if len(n_sylla)==4 and n_sylla[0] in (10, 11, 12, 13) and n_sylla[0]==n_sylla[1] and n_sylla[0]==n_sylla[3] and n_sylla[2] == n_sylla[0]+1:
                                        pass
                                    else:
                                        # В одной строке число слогов отличается от 3х других строк:
                                        #
                                        # У Лукоморья дуб зеленый,   - 9 слогов
                                        # Под ним живут русалки,     - 7 слогов
                                        # А в будках - покемоны,     - 7 слогов
                                        # Китайские пугалки.         - 7 слогов
                                        metre_defects_score *= 0.1

                    # Определяем рифмуемость
                    last_pwords = [pline[1].get_rhyming_tail() for pline in plinev]

                    rhyme_scheme, rhyme_score, rhyme_graph = self.detect_rhyming(last_pwords, rhyming_detection_cache)

                    # Учтем эффект ритма второго порядка: если рифмующиеся строки
                    # имеют точно совпадающие паттерны ударения, то уменьшим отличие скоров этих строк от 1 на 0.80.
                    #
                    # Например:
                    #
                    # Когда́ твоё́ чело́ избороздя́т
                    # Глубо́кими следа́ми со́рок зи́м,
                    # Кто бу́дет по́мнить ца́рственный наря́д,
                    # Гнуша́ясь жа́лким ру́бищем твои́м?
                    #
                    # Тут строки 1 и 3 совпадают в позициях ударений:
                    #
                    # 0101010001
                    # 0100010101
                    # 0101010001
                    # 0101010001

                    line_tscores = self.recalc_line_tscores(plinev, rhyme_graph)
                    #line_tscores = [pline[0].get_score() for pline in plinev]
                    #if rhyme_scheme == 'ABAB':
                    #    for i1, i2 in [(0, 2), (1, 3)]:
                    #        if plinev[i1][1].stress_signature == plinev[i2][1].stress_signature:
                    #            for j in [i1, i2]:
                    #                line_tscores[j] += (1.0 - line_tscores[j])*0.66
                    total_score = metre_defects_score * rhyme_score * mul(line_tscores)

                    if total_score > best_score:
                        best_score = total_score
                        best_metre = metre_name
                        best_rhyme_scheme = rhyme_scheme
                        best_variant = plinev
                        best_rhyme_graph = rhyme_graph

                        if best_score > self.early_stopping_threshold_score:
                            break

        # Если не получилось найти хороший маппинг с основными метрами, то пробуем дольники
        if best_score <= self.early_stopping_threshold_score and self.enable_dolnik:
            # Определим максимальное число слогов. Это нужно для дольников.
            num_syllables = [pline.get_num_syllables() for pline in plines]
            dolnik_patterns = self.get_dolnik_patterns(num_syllables)
            stressed_words_groups0 = [
                StressVariantsSlot.build(poetry_words=pline.pwords, aligner=self, allow_stress_shift=False, allow_unstress12=True) for
                pline in plines]

            for dolnik_pattern in dolnik_patterns:
                new_stress_lines = []
                for ipline, pline in enumerate(plines):
                    cursor = MetreMappingCursor(dolnik_pattern[ipline % len(dolnik_pattern)], prefix=0)

                    new_stress_line = None
                    best_mapping = None
                    max_score = 0.0

                    for metre_mapping in cursor.map(stressed_words_groups0[ipline], self):
                        if metre_mapping.score > max_score:
                            max_score = metre_mapping.score
                            stressed_words = [m.word for m in metre_mapping.word_mappings]
                            new_stress_line = LineStressVariant(pline, stressed_words, self)
                            best_mapping = metre_mapping

                    new_stress_lines.append((best_mapping, new_stress_line))

                # Определяем рифмуемость
                last_pwords = [line[1].get_rhyming_tail() for line in new_stress_lines]
                rhyme_scheme, rhyme_score, rhyme_graph = self.detect_rhyming(last_pwords, rhyming_detection_cache)

                total_score = rhyme_score * mul([pline[0].get_score() for pline in new_stress_lines])
                if total_score > best_score:
                    best_score = total_score
                    best_metre = 'dolnik'
                    best_rhyme_scheme = rhyme_scheme
                    best_variant = new_stress_lines
                    best_rhyme_graph = rhyme_graph

            # Некоторые четверостишия размечаются так, что в них классический и нерегулярный метр чередуются:
            # Голос яшмовой флейты слышу.
            # Кто же это играет на ней?
            # Ночь. Беседка. Луна все выше,
            # На душе все грустней и грустней.


        # 20-04-2025
        if best_score <= 0.1:
            score, stressed_lines, line_mappings, meter, rhyme_scheme, rhyme_graph = self.align_weak0(lines)

            do_prefer_weak = False
            if rhyme_scheme.count('-') < best_rhyme_scheme.count('-'):
                # Accept this alignment because it detects more rhymes
                do_prefer_weak = True
            else:
                if self.detect_stress_lacunas([v[0] for v in best_variant]):
                    do_prefer_weak = True

            if do_prefer_weak:
                return PoetryAlignment(stressed_lines,
                                       score,
                                       meter,
                                       rhyme_scheme=rhyme_scheme,
                                       rhyme_graph=rhyme_graph,
                                       metre_mappings=line_mappings)

        if best_variant is None:
            # В этом случае вернем результат с нулевым скором и особым текстом, чтобы
            # можно было вывести в лог строки с каким-то дефолтными
            return PoetryAlignment.build_no_rhyming_result([pline.get_first_stress_variants(self) for pline in plines])
        else:
            # Возвращаем найденный вариант разметки и его оценку
            best_lines = [v[1] for v in best_variant]
            metre_mappings = [v[0] for v in best_variant]
            return PoetryAlignment(best_lines, best_score, best_metre,
                                   rhyme_scheme=best_rhyme_scheme,
                                   rhyme_graph=best_rhyme_graph,
                                   metre_mappings=metre_mappings)

    def align_weak0(self, lines):
        for line in lines:
            tokens = tokenize(line)
            ntokens = len(tokens)
            if ntokens > self.max_words_per_line:
                raise ValueError('Line is too long @3146: max_words_per_line={} tokens={} line="{}"'.format(self.max_words_per_line, '|'.join(tokens), line))

        plines = [PoetryLine.build(line, self.udpipe, self.accentuator) for line in lines]

        # 08.11.2022 добавлена защита от взрыва числа переборов для очень плохих генераций.
        for pline in plines:
            if sum((pword.n_vowels>=1) for pword in pline.pwords) >= self.max_words_per_line:
                raise ValueError('Line is too long @3153: "{}"'.format(pline))

        n_sylla = tuple(l.get_num_syllables() for l in plines)

        best_score = 0.0
        best_metre = None
        best_rhyme_scheme = None
        best_variant = None
        best_rhyme_graph = None
        rhyming_detection_cache = dict()

        # Проверяем основные метры.
        for allow_stress_shift in [True]:  # [False, True]:
            if best_score > self.early_stopping_threshold_score:
                break

            # Заранее сгенерируем для каждого слова варианты спеллчека и ударения...
            stressed_words_groups = [StressVariantsSlot.build(poetry_words=pline.pwords, aligner=self, allow_stress_shift=allow_stress_shift, allow_unstress12=False) for pline in plines]

            check_meters = meters
            if n_sylla == (9, 8, 9, 2):
                # Порошки - только ямб
                check_meters = [('ямб', (0, 1))]

            # Для каждой строки перебираем варианты разметки и оставляем по ~2 варианта в каждом метре.
            for metre_name, metre_signature in check_meters:
                best_scores = dict()

                # В каждой строке перебираем варианты расстановки ударений.
                for ipline, pline in enumerate(plines):
                    best_scores[ipline] = dict()

                    for prefix in self.get_prefixes_for_meter(metre_signature):
                        cursor = MetreMappingCursor(metre_signature, prefix=prefix)
                        for metre_mapping in cursor.map(stressed_words_groups[ipline], self):
                            metre_mapping.enforce_intrinsic_word_accentuation()
                            stressed_words = [m.word for m in metre_mapping.word_mappings]
                            new_stress_line = LineStressVariant(pline, stressed_words, self)

                            if new_stress_line.get_rhyming_tail().is_ok():
                                tail_str = new_stress_line.get_rhyming_tail().__repr__()
                                score = metre_mapping.get_score()
                                if tail_str not in best_scores[ipline]:
                                    prev_score = -1e3
                                else:
                                    prev_score = best_scores[ipline][tail_str][0].get_score()
                                if score > prev_score:
                                    best_scores[ipline][tail_str] = (metre_mapping, new_stress_line)

                # Теперь для каждой исходной строки имеем несколько вариантов расстановки ударений.
                # Перебираем сочетания этих вариантов, проверяем рифмовку и оставляем лучший вариант для данной метра.
                stressed_lines2 = [list() for _ in range(len(best_scores))]
                for iline, items2 in best_scores.items():
                    stressed_lines2[iline].extend(items2.values())

                vvx = list(itertools.product(*stressed_lines2))
                for ivar, plinev in enumerate(vvx):
                    # plinev это набор из двух экземпляров кортежей (MetreMappingResult, LineStressVariant).

                    # Различные дефекты ритма
                    metre_defects_score = 1.0
                    # 11.12.2022 сдвиг одной строки на 1 позицию
                    nprefixa = collections.Counter(pline[0].prefix for pline in plinev)
                    if nprefixa.get(0) == 1 or nprefixa.get(1) == 1:
                        metre_defects_score *= 0.1

                    # Есть жанры, где в четырех строках должно быть 3 варианта слоговой длины.
                    # Примеры: порошки (9-8-9-2) и артишоки (11-9-11-2).
                    # Для них не штрафуем за 3 варианта длины!
                    n_sylla = tuple(l[1].poetry_line.get_num_syllables() for l in plinev)
                    if n_sylla not in [(11, 9, 11, 2), (9, 8, 9, 2)]:
                        nsyllaba = collections.Counter(len(pline[1].stress_signature) for pline in plinev)
                        if len(nsyllaba) > 2:
                            # Есть более 2 длин строк в слогах
                            metre_defects_score *= 0.1
                        else:
                            for nsyllab, num in nsyllaba.most_common():
                                if num == 3:
                                    # 03.05.2024 особый случай с рубаи. третья строка обычно отличается по числу слогов.
                                    if len(n_sylla)==4 and n_sylla[0] in (10, 11, 12, 13) and n_sylla[0]==n_sylla[1] and n_sylla[0]==n_sylla[3] and n_sylla[2] == n_sylla[0]+1:
                                        pass
                                    else:
                                        # В одной строке число слогов отличается от 3х других строк:
                                        metre_defects_score *= 0.1

                    # Определяем рифмуемость
                    last_pwords = [pline[1].get_rhyming_tail() for pline in plinev]

                    rhyme_scheme, rhyme_score, rhyme_graph = self.detect_rhyming(last_pwords, rhyming_detection_cache)
                    line_tscores = self.recalc_line_tscores(plinev, rhyme_graph)
                    total_score = metre_defects_score * rhyme_score * mul(line_tscores)

                    if total_score > best_score:
                        best_score = total_score
                        best_metre = metre_name
                        best_rhyme_scheme = rhyme_scheme
                        best_variant = plinev
                        best_rhyme_graph = rhyme_graph

                        if best_score > self.early_stopping_threshold_score:
                            break

        # Если не получилось найти хороший маппинг с основными метрами, то пробуем дольники
        if best_score <= self.early_stopping_threshold_score and self.enable_dolnik:
            # Определим максимальное число слогов. Это нужно для дольников.
            num_syllables = [pline.get_num_syllables() for pline in plines]
            dolnik_patterns = self.get_dolnik_patterns(num_syllables)
            stressed_words_groups0 = [
                StressVariantsSlot.build(poetry_words=pline.pwords, aligner=self, allow_stress_shift=False, allow_unstress12=False) for
                pline in plines]

            for dolnik_pattern in dolnik_patterns:
                new_stress_lines = []
                for ipline, pline in enumerate(plines):
                    cursor = MetreMappingCursor(dolnik_pattern[ipline % len(dolnik_pattern)], prefix=0)

                    new_stress_line = None
                    best_mapping = None
                    max_score = 0.0

                    for metre_mapping in cursor.map(stressed_words_groups0[ipline], self):
                        if metre_mapping.score > max_score:
                            max_score = metre_mapping.score
                            metre_mapping.enforce_intrinsic_word_accentuation()
                            stressed_words = [m.word for m in metre_mapping.word_mappings]
                            new_stress_line = LineStressVariant(pline, stressed_words, self)
                            best_mapping = metre_mapping

                    new_stress_lines.append((best_mapping, new_stress_line))

                # Определяем рифмуемость
                last_pwords = [line[1].get_rhyming_tail() for line in new_stress_lines]
                rhyme_scheme, rhyme_score, rhyme_graph = self.detect_rhyming(last_pwords, rhyming_detection_cache)

                total_score = rhyme_score * mul([pline[0].get_score() for pline in new_stress_lines])
                if total_score > best_score:
                    best_score = total_score
                    best_metre = 'dolnik'
                    best_rhyme_scheme = rhyme_scheme
                    best_variant = new_stress_lines
                    best_rhyme_graph = rhyme_graph

        # Возвращаем найденный вариант разметки и его оценку
        best_lines = [v[1] for v in best_variant]
        metre_mappings = [v[0] for v in best_variant]
        return best_score, best_lines, metre_mappings, best_metre, best_rhyme_scheme, best_rhyme_graph

    def recalc_line_tscores(self, plinev, rhyme_graph):
        line_tscores = [pline[0].get_score() for pline in plinev]

        # Если две строки ПОЛНОСТЬЮ совпадают по расстановке ударений, то снижаем для них штрафы за дефекты.
        # if rhyme_scheme == 'ABAB':
        #     for i1, i2 in [(0, 2), (1, 3)]:
        #         if plinev[i1][1].stress_signature == plinev[i2][1].stress_signature:
        #             for j in [i1, i2]:
        #                 line_tscores[j] += (1.0 - line_tscores[j]) * 0.66
        for i1, (edge, pline) in enumerate(zip(rhyme_graph, plinev)):
            if edge is not None:
                i2 = i1 + edge
                if plinev[i1][1].stress_signature == plinev[i2][1].stress_signature:
                    for j in [i1, i2]:
                        line_tscores[j] += (1.0 - line_tscores[j]) * 0.10

        return line_tscores

    def align_nonstandard_block(self, lines, check_rhymes=False):
        plines = [PoetryLine.build(line, self.udpipe, self.accentuator) for line in lines]

        # 08.11.2022 добавлена защита от взрыва числа переборов для очень плохих генераций.
        for pline in plines:
            if sum((pword.n_vowels>=1) for pword in pline.pwords) >= self.max_words_per_line:
                raise ValueError('Line is too long @2185: max_words_per_line={} "{}"'.format(self.max_words_per_line, pline))

        best_score = 0.0
        best_metre = None
        best_rhyme_scheme = None
        best_variant = None
        best_metre_signature = None
        best_rhyme_graph = None

        allow_stress_shift = True

        rhyming_detection_cache = dict()

        # Заранее сгенерируем для каждого слова варианты спеллчека и ударения...
        stressed_words_groups = [StressVariantsSlot.build(poetry_words=pline.pwords, aligner=self, allow_stress_shift=allow_stress_shift, allow_unstress12=True) for pline in plines]

        # 30.11.2023 еще одна защита от взрывного роста числа вариантов
        for stressed_words_group in stressed_words_groups:
            if stressed_words_group.count_variants()>50:
                raise ValueError('Too many alternatives of accentuation in line: "{}"'.format(str(stressed_words_groups)))

        if len(plines) <= 6:
            plines_first = plines
        else:
            # Сначала подберем метр для первых строк, а оставшиеся строки будем размечать уже с этим метром.
            plines_first = plines[:4]

        # Для каждой строки перебираем варианты разметки и оставляем по ~2 варианта в каждом метре.
        for metre_name, metre_signature in meters:
            best_scores = dict()

            # В каждой строке перебираем варианты расстановки ударений.
            for ipline, pline in enumerate(plines_first):
                best_scores[ipline] = dict()

                for prefix in [0,]: # self.get_prefixes_for_meter(metre_signature):
                    cursor = MetreMappingCursor(metre_signature, prefix=prefix)
                    for metre_mapping in cursor.map(stressed_words_groups[ipline], self):
                        stressed_words = [m.word for m in metre_mapping.word_mappings]
                        new_stress_line = LineStressVariant(pline, stressed_words, self)

                        if new_stress_line.get_rhyming_tail().is_ok():
                            tail_str = new_stress_line.get_rhyming_tail().__repr__()
                            score = metre_mapping.get_score()
                            if tail_str not in best_scores[ipline]:
                                prev_score = -1e3
                            else:
                                prev_score = best_scores[ipline][tail_str][0].get_score()
                            if score > prev_score:
                                best_scores[ipline][tail_str] = (metre_mapping, new_stress_line)

            # Теперь для каждой исходной строки имеем несколько вариантов расстановки ударений.
            # Перебираем сочетания этих вариантов, проверяем рифмовку и оставляем лучший вариант для данной метра.
            stressed_lines2 = [list() for _ in range(len(best_scores))]
            for iline, items2 in best_scores.items():
                stressed_lines2[iline].extend(items2.values())

            # дальше все нехорошо сделано, так как число вариантов будет запредельное.
            # надо переделать на какую-то динамическую схему подбора.
            # пока тупо ограничим перебор первыми вариантами.
            vvx = list(itertools.product(*stressed_lines2))[:1000]
            for ivar, plinev in enumerate(vvx):
                # plinev это набор из двух экземпляров кортежей (MetreMappingResult, LineStressVariant).

                # Различные дефекты ритма
                metre_defects_score = 1.0
                # 11.12.2022 сдвиг одной строки на 1 позицию
                nprefixa = collections.Counter(pline[0].prefix for pline in plinev)
                if nprefixa.get(0) == 1 or nprefixa.get(1) == 1:
                    metre_defects_score *= 0.1

                # Определяем рифмуемость
                #nlines = len(plinev)
                #rhyme_scheme = '-' * nlines
                last_pwords = [pline[1].get_rhyming_tail() for pline in plinev]
                rhyme_scheme, rhyming_score, rhyme_graph = self.detect_rhyming(last_pwords, rhyming_detection_cache)
                if not check_rhymes:
                    rhyming_score = 1.0

                line_tscores = self.recalc_line_tscores(plinev, rhyme_graph)
                total_score = metre_defects_score * mul(line_tscores) * rhyming_score
                if total_score > best_score:
                    best_score = total_score
                    best_metre = metre_name
                    best_rhyme_scheme = rhyme_scheme
                    best_rhyme_graph = rhyme_graph
                    best_variant = plinev
                    best_metre_signature = metre_signature

                    if best_score > self.early_stopping_threshold_score:
                        break

        if best_score <= self.early_stopping_threshold_score:
            # Определим максимальное число слогов. Это нужно для дольников.
            num_syllables = [pline.get_num_syllables() for pline in plines_first]
            dolnik_patterns = self.get_dolnik_patterns(num_syllables)
            stressed_words_groups0 = [
                StressVariantsSlot.build(poetry_words=pline.pwords, aligner=self, allow_stress_shift=False, allow_unstress12=True) for
                pline in plines]

            for dolnik_pattern in dolnik_patterns:
                new_stress_lines = []
                for ipline, pline in enumerate(plines_first):  # plines
                    cursor = MetreMappingCursor(dolnik_pattern[ipline % len(dolnik_pattern)], prefix=0)

                    new_stress_line = None
                    best_mapping = None
                    max_score = 0.0

                    for metre_mapping in cursor.map(stressed_words_groups0[ipline], self):
                        if metre_mapping.score > max_score:
                            max_score = metre_mapping.score
                            stressed_words = [m.word for m in metre_mapping.word_mappings]
                            new_stress_line = LineStressVariant(pline, stressed_words, self)
                            best_mapping = metre_mapping

                    new_stress_lines.append((best_mapping, new_stress_line))

                # Определяем рифмуемость
                last_pwords = [line[1].get_rhyming_tail() for line in new_stress_lines]
                rhyme_scheme, rhyming_score, rhyme_graph = self.detect_rhyming(last_pwords, rhyming_detection_cache)
                if not check_rhymes:
                    rhyming_score = 1.0

                line_tscores = self.recalc_line_tscores(new_stress_lines, rhyme_graph)
                total_score = rhyming_score * mul(line_tscores)

                if total_score > best_score:
                    best_score = total_score
                    best_metre = 'dolnik'
                    best_rhyme_scheme = rhyme_scheme
                    best_variant = new_stress_lines
                    best_rhyme_graph = rhyme_graph
                    best_metre_signature = dolnik_pattern


        # 20-04-2025
        if best_score <= 0.1:
            score, stressed_lines, line_mappings, meter, rhyme_scheme, rhyme_graph = self.align_weak0(lines)

            do_prefer_weak = False
            if rhyme_scheme.count('-') < best_rhyme_scheme.count('-'):
                # Accept this alignment because it detects more rhymes
                do_prefer_weak = True
            else:
                if self.detect_stress_lacunas([v[0] for v in best_variant]):
                    do_prefer_weak = True

            if do_prefer_weak:
                return PoetryAlignment(stressed_lines,
                                       score,
                                       meter,
                                       rhyme_scheme=rhyme_scheme,
                                       rhyme_graph=rhyme_graph,
                                       metre_mappings=line_mappings)


        if best_variant is None:
            # В этом случае вернем результат с нулевым скором и особым текстом, чтобы
            # можно было вывести в лог строки с каким-то дефолтными
            #return None, None, None  #PoetryAlignment.build_no_rhyming_result([pline.get_first_stress_variants(self) for pline in plines])
            return PoetryAlignment.build_no_rhyming_result([pline.get_first_stress_variants(self) for pline in plines])

        if len(plines) > len(plines_first):
            # Осталось разметить хвост
            best_variant = list(best_variant)
            best_rhyme_scheme = None

            # best_metre_signature = None
            # for metre_name, metre_signature in meters:
            #     if metre_name == best_metre:
            #         best_metre_signature = metre_signature
            #         break

            def supply_line_meter(iline):
                if best_metre == 'dolnik':
                    return best_metre_signature[iline % len(best_metre_signature)]
                else:
                    return best_metre_signature

            for ipline, pline in enumerate(plines[len(plines_first):], start=len(plines_first)):
                best_scores = dict()

                prefix = 0
                cursor = MetreMappingCursor(supply_line_meter(ipline), prefix=prefix)

                for metre_mapping in cursor.map(stressed_words_groups[ipline], self):
                    stressed_words = [m.word for m in metre_mapping.word_mappings]
                    new_stress_line = LineStressVariant(pline, stressed_words, self)

                    if new_stress_line.get_rhyming_tail().is_ok():
                        tail_str = new_stress_line.get_rhyming_tail().__repr__()
                        score = metre_mapping.get_score()
                        if tail_str not in best_scores:
                            prev_score = -1e3
                        else:
                            prev_score = best_scores[tail_str][0].get_score()
                        if score > prev_score:
                            best_scores[tail_str] = (metre_mapping, new_stress_line)

                best_line_alignment = None
                best_metre_mapping = None
                best_line_score = -1.0

                for metre_mapping, new_stress_line in best_scores.values():
                    if metre_mapping.get_score() > best_line_score:
                        best_line_score = metre_mapping.get_score()
                        best_line_alignment = new_stress_line
                        best_metre_mapping = metre_mapping

                best_variant.append((best_metre_mapping, best_line_alignment))
                best_score *= best_metre_mapping.get_score()

        # Возвращаем найденный вариант разметки и его оценку
        best_lines = [v[1] for v in best_variant]
        metre_mappings = [v[0] for v in best_variant]

        if len(best_rhyme_graph) != len(best_variant):
            # получение топологии рифмовки всего текста.
            last_pwords = [pline.get_rhyming_tail() for _, pline in best_variant]
            i_rhymed_with_j = []
            num_rhymes = 0
            for i, tail_i in enumerate(last_pwords[:-1]):
                i_rhyming = None
                for j, tail_j in enumerate(last_pwords[i+1:i+5], start=i+1):
                    r = self.check_rhyming(tail_i, tail_j)
                    if r:
                        i_rhyming = j - i
                        num_rhymes += 1
                        break
                i_rhymed_with_j.append(i_rhyming)
            # для последней строки просто добавляем отсутствие рифмовки вперед, так как впереди ничего нет.
            i_rhymed_with_j.append(None)
            best_rhyme_graph = i_rhymed_with_j

        return PoetryAlignment(best_lines, best_score, best_metre,
                               rhyme_scheme=best_rhyme_scheme,
                               rhyme_graph=best_rhyme_graph,
                               metre_mappings=metre_mappings)

    def align_nonstandard_blocks(self, lines):
        stressed_lines = []
        metre_mappings = []
        total_score = 1.0
        best_metre = None

        # Разобьем весь текст на блоки по границам пустых строк
        block_lines = []
        lines2 = lines + ['']
        for line in lines2:
            if len(line) == 0:
                if block_lines:
                    stressed_lines_i, metre_mappings_i, metre_i = self.align_nonstandard_block(lines)
                    best_metre = metre_i
                    stressed_lines.extend(stressed_lines_i)
                    metre_mappings.extend(metre_mappings_i)

                # добаваляем пустую строку, разделявшую блоки.
                stressed_lines.append(LineStressVariant.build_empty_line())
                metre_mappings.append(MetreMappingResult.build_for_empty_line())
                block_lines = []
            else:
                block_lines.append(line)

        return PoetryAlignment(stressed_lines, total_score, best_metre, rhyme_scheme='', metre_mappings=metre_mappings)

    def build_from_markup(self, text):
        lines = text.split('\n')

        plines = [PoetryLine.build_from_markup(line, self.udpipe) for line in lines]
        stressed_lines = [pline.get_first_stress_variants(self) for pline in plines]

        mapped_meter, mapping_score = self.map_meters(stressed_lines)
        score = mapping_score * reduce(lambda x, y: x * y, [l.get_score() for l in stressed_lines])
        rhyme_scheme = ''

        # 13.08.2022 определяем схему рифмовки
        if len(lines) == 2:
            rhyme_scheme = '--'
            claus1 = stressed_lines[0].get_rhyming_tail()
            claus2 = stressed_lines[1].get_rhyming_tail()
            if self.check_rhyming(claus1, claus2):
                rhyme_scheme = 'AA'
        elif len(lines) == 4:
            rhyme_scheme = '----'
            claus1 = stressed_lines[0].get_rhyming_tail()
            claus2 = stressed_lines[1].get_rhyming_tail()
            claus3 = stressed_lines[2].get_rhyming_tail()
            claus4 = stressed_lines[3].get_rhyming_tail()

            r12 = self.check_rhyming(claus1, claus2)
            r13 = self.check_rhyming(claus1, claus3)
            r14 = self.check_rhyming(claus1, claus4)
            r23 = self.check_rhyming(claus2, claus3)
            r24 = self.check_rhyming(claus2, claus4)
            r34 = self.check_rhyming(claus3, claus4)

            if r12 and r23 and r34:
                rhyme_scheme = 'AAAA'
            elif r13 and r24 and not r12 and not r34:
                rhyme_scheme = 'ABAB'
            elif r12 and not r23 and r34:
                rhyme_scheme = 'AABB'
            elif r12 and not r23 and not r23 and r14:
                rhyme_scheme = 'AABA'
            elif r14 and r23 and not r12 and not r34:
                rhyme_scheme = 'ABBA'
            elif not r12 and r13 and not r34:
                rhyme_scheme = 'A-A-'
            elif not r12 and r24 and not r23:
                rhyme_scheme = '-A-A'

        return PoetryAlignment(stressed_lines, score, mapped_meter, rhyme_scheme=rhyme_scheme, metre_mappings=None)

    def detect_repeating(self, alignment, strict=False):
        if alignment.inner_alignments:
            for ia in alignment.inner_alignments:
                r = self.detect_repeating(ia, strict)
                if r:
                    return True

            return False
        else:
            # Повтор последних слов в разных строках
            last_words = [pline.get_rhyming_tail().stressed_word.form.lower() for pline in alignment.poetry_lines]
            for i1, word1 in enumerate(last_words):
                for word2 in last_words[i1+1:]:
                    if word1 == word2:
                        return True

            return self.detect_repeating_in_line(alignment, strict)

    def detect_repeating_in_line(self, alignment, strict=False):
        # Иногда генеративная модель выдает повторы существительных типа "любовь и любовь" в одной строке.
        # Такие генерации выглядят криво.
        # Данный метод детектирует повтор леммы существительного в строке.
        # 22.10.2022 добавлен учет глаголов и прилагательных
        for pline in alignment.poetry_lines:
            if pline.poetry_line.pwords:
                n_lemmas = collections.Counter()
                for pword in pline.poetry_line.pwords:
                    if pword.upos in ('NOUN', 'PROPN', 'ADJ', 'VERB'):
                        n_lemmas[pword.lemma] += 1
                    elif strict and pword.upos in ('ADV', 'PRON', 'SYM'):
                        n_lemmas[pword.lemma] += 1
                if n_lemmas and n_lemmas.most_common(1)[0][1] > 1:
                    return True

                if strict:
                    # Повтор слова длиннее 4 букв тоже считаем плохим
                    n_forms = collections.Counter()
                    for pword in pline.poetry_line.pwords:
                        if len(pword.form) >= 5:
                            n_forms[pword.upos + ':' + pword.form.lower().replace('ё', 'е')] += 1
                    if n_forms and n_forms.most_common(1)[0][1] > 1:
                        return True

                # любой повтор XXX XXX
                for w1, w2 in zip(pline.poetry_line.pwords, pline.poetry_line.pwords[1:]):
                    if w1.form.lower() == w2.form.lower() and w1.form[0] not in '.!?':
                        return True

                # также штрафуем за паттерн "XXX и XXX"
                for w1, w2, w3 in zip(pline.poetry_line.pwords, pline.poetry_line.pwords[1:], pline.poetry_line.pwords[2:]):
                    if w2.form.replace('\u0301', '') in ('и', ',', 'или', 'иль', 'аль', 'да'):
                        if w1.form.replace('\u0301', '').lower() == w3.form.replace('\u0301', '').lower() and w1.form.replace('\u0301', '') not in 'вновь еще ещё снова опять дальше ближе сильнее слабее сильней слабей тише'.split(' '):
                            return True

                # 01-11-2022 повтор формы существительного
                nouns = collections.Counter(w.form.lower() for w in pline.poetry_line.pwords if w.upos in ('NOUN', 'PROPN'))
                if len(nouns) > 0:
                    if nouns.most_common(1)[0][1] > 1:
                        return True

                # 01-11-2022 наличие в одной строке вариантов с "ё" и с "е" считаем повтором
                forms = [w.form.lower() for w in pline.poetry_line.pwords]
                for v1, v2 in [('поёт', 'поет')]:
                    if v1 in forms and v2 in forms:
                        return True

                # 14.08.2023 попадаются повторы из личной формы глагола и деепричастия:
                # Когда лишь любишь ты любя.
                #            ^^^^^^^^^^^^^^
                #print('DEBUG@1992')
                vlemmas = set()
                conv_lemmas = set()
                for w in pline.poetry_line.pwords:
                    if w.upos == 'VERB':
                        #print('DEBUG@1997 w.form=', w.form,  w.lemma)
                        if w.get_attr('VerbForm') == 'Conv':
                            conv_lemmas.add(w.lemma)
                        else:
                            vlemmas.add(w.lemma)

                if any((v in conv_lemmas) for v in vlemmas):
                    #print('DEBUG@2004')
                    return True

        return False

    def detect_poor_poetry(self, alignment):
        """Несколько эвристик для обнаружения скучных рифм, которые мы не хотим получать"""

        if alignment.inner_alignments:
            for ia in alignment.inner_alignments:
                r = self.detect_poor_poetry(ia)
                if r:
                    return True

            return False
        else:
            last_words = [pline.get_rhyming_tail().stressed_word for pline in alignment.poetry_lines]
            last_words = [(x.form.lower() if x else '') for x in last_words ]

            # Проверяем банальный повтор слова
            # 18.08.2023 проверяем случаи "легко-нелегко"
            # 09.11.2023 ограничиваемся 5 строками для поиска повтора.
            for i1, word1 in enumerate(last_words[:-1]):
                for word2 in last_words[i1+1: i1+6]:
                    form1 = word1.lower().replace('ё', 'е')
                    form2 = word2.lower().replace('ё', 'е')
                    if form1 and form1 == form2:
                        return True
                    if form1 and form1 == 'не'+form2 or 'не'+form1 == form2:
                        return True

            # Если два глагольных окончания, причем одно является хвостом другого - это бедная рифма:
            # ждать - подождать
            # смотреть - посмотреть
            # etc.
            rhyme_pairs = []
            if alignment.rhyme_scheme in ('ABAB', 'A-A-', '-A-A'):
                rhyme_pairs.append((alignment.poetry_lines[0].get_rhyming_tail(), alignment.poetry_lines[2].get_rhyming_tail()))
                rhyme_pairs.append((alignment.poetry_lines[1].get_rhyming_tail(), alignment.poetry_lines[3].get_rhyming_tail()))
            elif alignment.rhyme_scheme == 'ABBA':
                rhyme_pairs.append((alignment.poetry_lines[0].get_rhyming_tail(), alignment.poetry_lines[3].get_rhyming_tail()))
                rhyme_pairs.append((alignment.poetry_lines[1].get_rhyming_tail(), alignment.poetry_lines[2].get_rhyming_tail()))
            elif alignment.rhyme_scheme == 'AABA':
                rhyme_pairs.append((alignment.poetry_lines[0].get_rhyming_tail(), alignment.poetry_lines[1].get_rhyming_tail()))
                rhyme_pairs.append((alignment.poetry_lines[0].get_rhyming_tail(), alignment.poetry_lines[3].get_rhyming_tail()))
            elif alignment.rhyme_scheme == 'AABB':
                rhyme_pairs.append((alignment.poetry_lines[0].get_rhyming_tail(), alignment.poetry_lines[1].get_rhyming_tail()))
                rhyme_pairs.append((alignment.poetry_lines[2].get_rhyming_tail(), alignment.poetry_lines[3].get_rhyming_tail()))
            elif alignment.rhyme_scheme in ('AAAA', '----'):
                rhyme_pairs.append((alignment.poetry_lines[0].get_rhyming_tail(), alignment.poetry_lines[1].get_rhyming_tail()))
                rhyme_pairs.append((alignment.poetry_lines[1].get_rhyming_tail(), alignment.poetry_lines[2].get_rhyming_tail()))
                rhyme_pairs.append((alignment.poetry_lines[2].get_rhyming_tail(), alignment.poetry_lines[3].get_rhyming_tail()))

            for tail1, tail2 in rhyme_pairs:
                word1 = tail1.stressed_word
                word2 = tail2.stressed_word

                form1 = word1.poetry_word.form.lower()
                form2 = word2.poetry_word.form.lower()

                if word1.poetry_word.upos == 'VERB' and word2.poetry_word.upos == 'VERB':
                    # 11-01-2022 если пара слов внесена в специальный список рифмующихся слов, то считаем,
                    # что тут все нормально:  ВИТАЮ-ТАЮ
                    if (form1, form2) in self.accentuator.rhymed_words:
                        continue

                    #if any((form1.endswith(e) and form2.endswith(e)) for e in 'ли ла ло л м шь т тся у те й ю ь лись лась лось лся тся ться я шись в'.split(' ')):
                    if any((form1.endswith(e) and form2.endswith(e)) for e in 'ли ла ло л тся те лись лась лось лся тся ться шись'.split(' ')):
                        return True

                    if form1.endswith(form2):
                        return True
                    elif form2.endswith(form1):
                        return True

                    # Попробуем поймать повторы однокоренных глаголов типа УДЕРЖАТЬ-ЗАДЕРЖАТЬ
                    if form1 in self.word_segmentation and form2 in self.word_segmentation:
                        segm1 = self.word_segmentation[form1]
                        segm2 = self.word_segmentation[form2]

                        if segm1.without_prefix == segm2.without_prefix:
                            return True

                # Попробуем поймать рифмовку форм слова, когда отличаются только окончания: тыквы-тыква
                if word1.poetry_word.upos in ('NOUN', 'PROPN', 'ADJ') and word1.poetry_word.upos == word2.poetry_word.upos:
                    if word1.poetry_word.lemma == word2.poetry_word.lemma:
                        return True

            # Для других частей речи проверяем заданные группы слов.
            for bad_group in ['твой мой свой'.split(' '),
                              'тебе мне себе'.split(' '),
                              'него его'.split(' '),
                              'твои свои'.split(' '),
                              'наши ваши'.split(' '),
                              'меня тебя себя'.split(' '),
                              'мной тобой собой'.split(' '),
                              'мною тобой собою'.split(' '),
                              'нее ее неё её'.split(' '),
                              '~шел ~шёл'.split(' '),
                              'твоем твоём своем своём моем моём'.split(' '),
                              'когда никогда навсегда кое-когда'.split(' '),
                              'кто никто кое-кто'.split(' '),
                              'где нигде везде'.split(' '),
                              'каких никаких таких сяких'.split(' '),
                              'какого никакого такого сякого'.split(' '),
                              'какую никакую такую сякую'.split(' '),
                              #'сможем поможем можем'.split(' '),
                              'смогу помогу могу'.split(' '),
                              #'поехал уехал наехал приехал въехал заехал доехал подъехал'.split(' '),
                              'того чего ничего никого кого'.split(' '),
                              #'ждали ожидали'.split(' '),
                              #'подумать придумать думать продумать надумать удумать'.split(' '),
                              'всегда навсегда'.split(' ')
                              ]:
                if bad_group[0][0] == '~':
                    n_hits = 0
                    for last_word in last_words:
                        n_hits += sum((re.match('^.+' + ending[1:] + '$', last_word, flags=re.I) is not None) for ending in bad_group)
                else:
                    n_hits = sum((word in last_words) for word in bad_group)

                if n_hits > 1:
                    return True

            return False

    def detect_rhyme_repeatance(self, alignment):
        """Обнаруживаем повтор слова в рифмовке"""
        if alignment.inner_alignments:
            for ia in alignment.inner_alignments:
                r = self.detect_rhyme_repeatance(ia)
                if r:
                    return True

            return False
        else:
            last_words = [pline.get_rhyming_tail().stressed_word.form.lower() for pline in alignment.poetry_lines]

            # Проверяем банальный повтор слова
            for i1, word1 in enumerate(last_words[:-1]):
                for word2 in last_words[i1+1:]:
                    form1 = word1.lower().replace('ё', 'е')
                    form2 = word2.lower().replace('ё', 'е')
                    if form1 == form2:
                        return True

            return False

    def analyze_defects(self, alignment):
        defects = Defects()

        # Анализируем количество слогов в строках
        nvs = [count_vowels(line.get_unstressed_line()) for line in alignment.poetry_lines]

        nv1 = nvs[0]
        nv_deltas = [(nv - nv1) for nv in nvs]

        if len(set(nv_deltas)) != 1:
            # Не все строки имеют равное число слогов. Выясняем подробности.
            are_good_nvs = False

            if alignment.rhyme_scheme in ('ABAB', 'A-A-', '-A-A'):
                if nv_deltas[0] == nv_deltas[2] and nv_deltas[1] == nv_deltas[3]:
                    delta = abs(nv_deltas[0] - nv_deltas[2])  # чем сильнее отличается число шагов, тем больше штраф
                    defects.add_defect(Defect(penalty=0.95*delta, description='@1858'))
                    are_good_nvs = True
                elif nv_deltas[0] == nv_deltas[1] and nv_deltas[1] == nv_deltas[2]:
                    # Первые три строки имеют равно число слогов, а последняя - отличается
                    delta = abs(nv_deltas[0] - nv_deltas[3])  # чем сильнее отличается число шагов, тем больше штраф
                    defects.add_defect(Defect(penalty=0.90*delta, description='@1863'))
                    are_good_nvs = True
            elif alignment.rhyme_scheme in ('ABBA', '-AA-'):
                if (nv_deltas[0] == nv_deltas[3]) and (nv_deltas[1] == nv_deltas[2]):
                    delta = abs(nv_deltas[0] - nv_deltas[1])
                    defects.add_defect(Defect(penalty=0.90 * delta, description='@1868'))
                    are_good_nvs = True
            else:
                nvz = collections.Counter(nv_deltas)
                if len(nvz) == 2:
                    m = nvz.most_common()
                    delta = max(nv_deltas) - min(nv_deltas)
                    if m[0][1] == 2 and m[1][1] == 2:
                        defects.add_defect(Defect(penalty=0.85 * delta, description='@1876'))
                        are_good_nvs = True
                    elif nv_deltas[3] == m[1][0]:
                        # Последняя строка в катрене имеет отличное число слогов от трех других.
                        defects.add_defect(Defect(penalty=0.80 * delta, description='@1880'))
                        are_good_nvs = True

            if not are_good_nvs:
                defects.add_defect(Defect(0.5, description='@1884'))

        return defects

    def markup_song_line(self, line):
        if len(line) == 0:
            return []

        opt_words = ['лишь', 'вроде', 'если', 'чтобы', 'когда', 'просто', 'мимо', 'даже', 'всё', 'хотя', 'едва', 'нет',
                     'эти', 'эту', 'это', 'мои', 'твои', 'моих', 'твоих', 'моим', 'твоим', 'моей', 'твоей',
                     'мою', 'твою', 'его', 'ее', 'её', 'себе', 'тебя', 'свою', 'свои', 'своим', 'они', 'она',
                     'уже', 'есть', 'раз', 'быть']
        res_tokens = []
        parsings = self.udpipe.parse_text(line)
        if parsings is None:
            raise RuntimeError()

        for parsing in parsings:
            # найдем индекс токена в конце строки, который будет рифмоваться.
            itoken = len(parsing)-1
            rhyming_token_index = 1000000
            while itoken > 0:
                if re.match(r'^\w+$', parsing[itoken].form) is not None and count_vowels(parsing[itoken].form) > 0:
                    rhyming_token_index = itoken + 1
                    break
                else:
                    itoken -= 1

            for itoken, ud_token in enumerate(parsing, start=1):
                stress_pos = 0
                #word = ud_token.form.lower()
                word = self.accentuator.yoficate(ud_token.form.lower())

                nvowels = sum((c in 'уеыаоэёяию') for c in word)

                if ud_token.upos in ('PRON', 'ADV', 'DET') and nvowels == 1:
                    # Односложные наречия, местоимения и т.д.
                    is_optional_stress = True
                elif ud_token.upos in ('PUNCT', 'ADP', 'PART', 'SCONJ', 'CCONJ', 'INTJ'):
                    is_optional_stress = True
                elif word in opt_words:
                    is_optional_stress = True
                else:
                    is_optional_stress = False

                if not is_optional_stress or itoken == rhyming_token_index:
                    stress_pos = self.accentuator.get_accent(word, ud_tags=ud_token.tags + [ud_token.upos])

                cx = []
                vowel_counter = 0
                for c in ud_token.form:
                    cx.append(c)
                    if c.lower() in 'уеыаоэёяию':
                        vowel_counter += 1
                        if vowel_counter == stress_pos:
                            cx.append('\u0301')
                token2 = ''.join(cx)

                form2 = word
                if ud_token.form[0].upper() == ud_token.form[0]:
                    form2 = word[0].upper() + word[1:]

                res_tokens.append({'form': form2,
                                   'lemma': ud_token.lemma,
                                   'upos': ud_token.upos,
                                   'tags': ud_token.tags,
                                   'stress_pos': stress_pos,
                                   'vowel_count': vowel_counter,
                                   'rendition': token2})

        return res_tokens

    def get_rhyming_tail(self, rap_tokens):
        while rap_tokens:
            if re.search(r'\w', rap_tokens[-1]['form']) is not None:
                return rap_tokens[-1]
            rap_tokens = rap_tokens[:-1]
        return None

    def align_song(self, song_lines: List[str], genre: str) -> SongAlignment:
        blocks = []
        lines = [line.strip() for line in song_lines]
        block = []
        while lines:
            if lines[0]:
                block.append(lines[0])
            else:
                if block:
                    blocks.append(block)
                block = []
            lines = lines[1:]

        if block:
            blocks.append(block)

        song_alignment = SongAlignment()
        for block in blocks:
            block_title = None
            if re.search(r'^Припев[.:]', block[0]) is not None:
                block_title = block[0]
                block = block[1:]

            if block:
                try:
                    markup = self.align(block, check_rhymes=True)
                except Exception as ex:
                    poetry_lines = [self.markup_song_line(line) for line in block]
                    markup = PoetryAlignment.build_no_rhyming_result(poetry_lines)
            else:
                markup = None

            block_alignment = SongAlignmentBlock(block_title, markup)
            song_alignment.blocks.append(block_alignment)

        return song_alignment

    def markup_prose(self, text):
        res_lines = []
        no_accent_words = ['а', 'и', 'или', 'но', 'не', 'ни', 'же', 'ли', 'бы', 'ка', 'по', 'у', 'об', 'со', 'за']
        opt_words = ['лишь', 'вроде', 'если', 'чтобы', 'когда', 'просто', 'мимо', 'даже', 'всё', 'хотя', 'едва', 'нет',
                     'эти', 'эту', 'это', 'мои', 'твои', 'моих', 'твоих', 'моим', 'твоим', 'моей', 'твоей',
                     'мою', 'твою', 'его', 'ее', 'её', 'себе', 'тебя', 'свою', 'свои', 'своим', 'они', 'она',
                     'уже', 'есть', 'раз', 'быть', 'для', 'была', 'были', 'было']

        for line in text.split('\n'):
            res_tokens = []

            # Отбиваем некоторые символы пунктуации пробелами, чтобы они гарантировано не слиплись со словом
            # в токенизаторе UDPipe/Stanza.
            line2 = line
            for c in '\'.‚,?!:;…-–—«»″”“„‘’`ʹ"˝[]‹›·<>*/=()+®©‛¨×№\u05f4':
                line2 = line2.replace(c, ' ' + c + ' ').replace('  ', ' ')

            parsings = self.udpipe.parse_text(line2)
            if parsings is None:
                raise RuntimeError()

            for parsing in parsings:
                iword = 0
                nwords = len(parsing)
                while iword < nwords:
                    ud_token = parsing[iword]

                    # Если в слове явно задано ударение символом \u0301...
                    if '\u0301' in ud_token.form:
                        res_tokens.append(ud_token.form)
                        iword += 1
                        continue

                    token2 = parsing[iword+1] if iword < nwords-1 else None
                    token3 = parsing[iword+2] if iword < nwords-2 else None

                    stress_pos = 0
                    word = ud_token.form.lower()
                    word2 = token2.form.lower() if token2 else ''
                    word3 = token3.form.lower() if token3 else ''
                    nvowels = sum((c in 'уеыаоэёяию') for c in word)

                    key = (word, word2, word3)
                    collocs = self.collocations.get(key, None)
                    if collocs is not None:
                        # Если есть варианты ударения в словосочетании, то выбираем случайно один из них
                        colloc = random.choice(collocs)

                        if colloc.stressed_word_index == 0:
                            # первое слово становится ударным, второе и третье - безударные
                            stress_pos = self.accentuator.get_accent(word, ud_tags=ud_token.tags + [ud_token.upos])
                            aw = render_word_with_stress(ud_token.form, stress_pos)
                            res_tokens.append(aw)
                            res_tokens.append(token2.form)
                            res_tokens.append(token3.form)
                            iword += 3
                        elif colloc.stressed_word_index == 1:
                            # первое слово становится безударным, второе - ударное, третье - безударное
                            res_tokens.append(ud_token.form)

                            stress_pos = self.accentuator.get_accent(word2, ud_tags=token2.tags + [token2.upos])
                            aw = render_word_with_stress(token2.form, stress_pos)
                            res_tokens.append(aw)

                            res_tokens.append(token3.form)

                            iword += 3
                        else:
                            # первое и второе слово безударные, третье - ударное
                            res_tokens.append(ud_token.form)
                            res_tokens.append(token2.form)

                            stress_pos = self.accentuator.get_accent(word3, ud_tags=token3.tags + [token3.upos])
                            aw = render_word_with_stress(token3.form, stress_pos)
                            res_tokens.append(aw)

                            iword += 3
                        continue

                    key = (word, word2)
                    collocs = self.collocations.get(key, None)
                    if collocs is not None:
                        # Если есть варианты ударения в словосочетании, то выбираем случайно один из них
                        colloc  = random.choice(collocs)
                        if colloc.stressed_word_index == 0:
                            # первое слово становится ударным, второе - безударное
                            stress_pos = self.accentuator.get_accent(word, ud_tags=ud_token.tags + [ud_token.upos])
                            aw = render_word_with_stress(ud_token.form, stress_pos)
                            res_tokens.append(aw)
                            res_tokens.append(token2.form)

                            iword += 2
                        elif colloc.stressed_word_index == 1:
                            # первое слово становится безударным, второе - ударное
                            res_tokens.append(ud_token.form)

                            stress_pos = self.accentuator.get_accent(word2, ud_tags=token2.tags + [token2.upos])
                            aw = render_word_with_stress(token2.form, stress_pos)
                            res_tokens.append(aw)

                            iword += 2
                        continue

                    # для слов из списка ударение опционально, его мы будем выставлять рандомно, т.е. ставить или нет,
                    # с вероятностью 0.5
                    is_optional_stress = False

                    # некоторые слова никогда не будем делать ударными (за исключением случаев из таблицы словосочетаний)
                    supress_accent = False

                    if word in no_accent_words:
                        supress_accent = True
                    elif ud_token.upos in ('PRON', 'ADV', 'DET') and nvowels == 1:
                        # Односложные наречия, местоимения и т.д.
                        is_optional_stress = True
                    elif ud_token.upos == 'ADP' and count_vowels(word) <= 1:
                        supress_accent = True
                        is_optional_stress = False
                    elif ud_token.upos == 'PART' and word in ['и', 'а', 'же', 'ка', 'то', 'ли', 'бы', 'уж', 'не', 'но', 'ни', 'аж']:
                        supress_accent = True
                        is_optional_stress = False
                    elif ud_token.upos in ('PUNCT', 'ADP', 'PART', 'SCONJ', 'CCONJ', 'INTJ'):
                        is_optional_stress = True
                    elif word in opt_words:
                        is_optional_stress = True

                    if supress_accent:
                        stress_pos = -1
                    elif word in self.accentuator.ambiguous_accents2:
                        # для слов типа пОнял-понЯл, где есть варианты ударения для одной формы,
                        # берем первый вариант в 70% случаев, а в 30% - второй. Подразумевается,
                        # что в словаре ambiguous_accents2.json первый вариант ударения - основной.

                        if is_optional_stress and random.random() < 0.3:
                            # несколько слов типа "была" могут вообще быть без ударения
                            stress_pos = -1
                        else:
                            stress_pos = self.accentuator.ambiguous_accents2[word][0] if random.random() <= 0.70 else self.accentuator.ambiguous_accents2[word][1]
                            if stress_pos == -1:
                                raise RuntimeError()
                    elif not is_optional_stress:
                        stress_pos = self.accentuator.get_accent(word, ud_tags=ud_token.tags + [ud_token.upos])
                    else:
                        if random.random() > 0.5:
                            stress_pos = self.accentuator.get_accent(word, ud_tags=ud_token.tags + [ud_token.upos])

                    if stress_pos == -1:
                        token2 = ud_token.form
                    else:
                        token2 = render_word_with_stress(ud_token.form, stress_pos)

                    res_tokens.append(token2)

                    iword += 1

            res_lines.append(normalize_whitespaces(' '.join(res_tokens)))

        markup = '\n'.join(res_lines)

        # Найдем все случаи, когда два слова разделены дефисом без пробелов, чтобы восстановить эту ситуацию после разметки.
        w1_defis_w2_list = set(re.findall(r'([\w\d]+)-([\w\d]+)', text))

        # Также найдем все случаи, когда два числа разделены запятой без пробела, чтобы восстановить такой паттерн после разметки
        d_comma_d = set(re.findall(r'(\d+)([,.])(\d+)', text))

        markup = normalize_whitespaces(markup)

        def remove_stress(s):
            return s.replace('\u0301', '')

        markup = re.sub(r'([\w\d\u0301]+) - ([\w\d\u0301]+)', lambda m: m.group(1) + '-' + m.group(2) if (remove_stress(
            m.group(1)), remove_stress(m.group(2))) in w1_defis_w2_list else m.group(0), markup)

        markup = re.sub(r'(\d+)([.,]) (\d+)', lambda m: m.group(1) + m.group(2) + m.group(3) if (m.group(1), m.group(2),
                                                                                                 m.group(
                                                                                                     3)) in d_comma_d else m.group(
            0), markup)

        return markup

    def align_rap(self, rap_text: str) -> SongAlignment:
        #raise NotImplementedError()
        #return self.align_song(rap_text.split('\n'), genre='рэп')

        no_accent_words = ['а', 'и', 'или', 'но', 'не', 'ни', 'же', 'ли', 'бы', 'ка', 'по', 'у', 'об', 'со', 'за']
        opt_words = ['лишь', 'вроде', 'если', 'чтобы', 'когда', 'просто', 'мимо', 'даже', 'всё', 'хотя', 'едва', 'нет',
                     'эти', 'эту', 'это', 'мои', 'твои', 'моих', 'твоих', 'моим', 'твоим', 'моей', 'твоей',
                     'мою', 'твою', 'его', 'ее', 'её', 'себе', 'тебя', 'свою', 'свои', 'своим', 'они', 'она',
                     'уже', 'есть', 'раз', 'быть', 'для', 'была', 'были', 'было']

        aligned_blocks = []
        for block_text in rap_text.split('\n\n'):
            block_header = None
            block_lines = None
            lines = block_text.split('\n')
            if lines:
                block_header = None
                block_lines = list(lines)
                if re.match(r'^(Припев\.|Припев:|Бридж:|Куплет:)$', lines[0]):
                    block_header = lines[0]
                    block_lines = block_lines[1:]

                stressed_lines = []

                for line in block_lines:
                    res_items = []

                    # Отбиваем некоторые символы пунктуации пробелами, чтобы они гарантировано не слиплись со словом
                    # в токенизаторе UDPipe/Stanza.
                    line2 = line
                    for c in '\'.‚,?!:;…-–—«»″”“„‘’`ʹ"˝[]‹›·<>*/=()+®©‛¨×№\u05f4':
                        line2 = line2.replace(c, ' ' + c + ' ').replace('  ', ' ')

                    parsings = self.udpipe.parse_text(line2)
                    if parsings is None:
                        raise RuntimeError()

                    for parsing in parsings:
                        iword = 0
                        nwords = len(parsing)
                        while iword < nwords:
                            ud_token = parsing[iword]

                            # Если в слове явно задано ударение символом \u0301...
                            #if '\u0301' in ud_token.form:
                            #    res_tokens.append(ud_token.form)
                            #    iword += 1
                            #    continue

                            token2 = parsing[iword + 1] if iword < nwords - 1 else None
                            token3 = parsing[iword + 2] if iword < nwords - 2 else None

                            stress_pos = 0
                            word = ud_token.form.lower()
                            word2 = token2.form.lower() if token2 else ''
                            word3 = token3.form.lower() if token3 else ''
                            nvowels = sum((c in 'уеыаоэёяию') for c in word)

                            key = (word, word2, word3)
                            collocs = self.collocations.get(key, None)
                            if collocs is not None:
                                item = TokenStressVariants()
                                for colloc in collocs:

                                    if colloc.stressed_word_index == 0:
                                        # первое слово становится ударным, второе и третье - безударные
                                        g = TokenGroup()

                                        accentuation = self.accentuator.get_accents(word, ud_tags=ud_token.tags + [ud_token.upos])
                                        pword1 = PoetryWord(ud_token.lemma, ud_token.form, ud_token.upos, ud_token.tags, accentuation)
                                        g.add(pword1)

                                        pword2 = PoetryWord(token2.lemma, token2.form, token2.upos, token2.tags, [WordAccentuation.build_nostress()])
                                        g.add(pword2)

                                        pword3 = PoetryWord(token3.lemma, token3.form, token3.upos, token3.tags, [WordAccentuation.build_nostress()])
                                        g.add(pword3)

                                        item.add(g)
                                    elif colloc.stressed_word_index == 1:
                                        # первое слово становится безударным, второе - ударное, третье - безударное
                                        g = TokenGroup()

                                        pword1 = PoetryWord(ud_token.lemma, ud_token.form, ud_token.upos, ud_token.tags, [WordAccentuation.build_nostress()])
                                        g.add(pword1)

                                        accentuation = self.accentuator.get_accents(word2, ud_tags=token2.tags + [token2.upos])
                                        pword2 = PoetryWord(token2.lemma, token2.form, token2.upos, token2.tags, accentuation)
                                        g.add(pword2)

                                        pword3 = PoetryWord(token3.lemma, token3.form, token3.upos, token3.tags, [WordAccentuation.build_nostress()])
                                        g.add(pword3)

                                        item.add(g)
                                    elif colloc.stressed_word_index == 2:
                                        # первое и второе слово безударные, третье - ударное
                                        g = TokenGroup()

                                        pword1 = PoetryWord(ud_token.lemma, ud_token.form, ud_token.upos, ud_token.tags, [WordAccentuation.build_nostress()])
                                        g.add(pword1)

                                        pword2 = PoetryWord(token2.lemma, token2.form, token2.upos, token2.tags, [WordAccentuation.build_nostress()])
                                        g.add(pword2)

                                        accentuation = self.accentuator.get_accents(word3, ud_tags=token3.tags + [token3.upos])
                                        pword3 = PoetryWord(token3.lemma, token3.form, token3.upos, token3.tags, accentuation)
                                        g.add(pword3)

                                        item.add(g)
                                    else:
                                        raise RuntimeError()

                                iword += 3
                                res_items.append(item)
                                continue

                            key = (word, word2)
                            collocs = self.collocations.get(key, None)
                            if collocs is not None:
                                item = TokenStressVariants()
                                for colloc in collocs:
                                    if colloc.stressed_word_index == 0:
                                        # первое слово становится ударным, второе - безударное
                                        g = TokenGroup()

                                        accentuation = self.accentuator.get_accents(word, ud_tags=ud_token.tags + [ud_token.upos])
                                        pword1 = PoetryWord(ud_token.lemma, ud_token.form, ud_token.upos, ud_token.tags, accentuation)
                                        g.add(pword1)

                                        pword2 = PoetryWord(token2.lemma, token2.form, token2.upos, token2.tags, [WordAccentuation.build_nostress()])
                                        g.add(pword2)

                                        item.add(g)
                                    elif colloc.stressed_word_index == 1:
                                        # первое слово становится безударным, второе - ударное
                                        g = TokenGroup()

                                        pword1 = PoetryWord(ud_token.lemma, ud_token.form, ud_token.upos, ud_token.tags, [WordAccentuation.build_nostress()])
                                        g.add(pword1)

                                        accentuation = self.accentuator.get_accents(word2, ud_tags=token2.tags + [token2.upos])
                                        pword2 = PoetryWord(token2.lemma, token2.form, token2.upos, token2.tags, accentuation)
                                        g.add(pword2)

                                        item.add(g)
                                    else:
                                        msg = 'ERROR@3257 Pair of word without stress in "collocations": {}'.format(colloc)
                                        raise RuntimeError(msg)

                                res_items.append(item)
                                iword += 2
                                continue

                            # для слов из списка ударение опционально, его мы будем выставлять рандомно, т.е. ставить или нет,
                            # с вероятностью 0.5
                            is_optional_stress = False

                            # некоторые слова никогда не будем делать ударными (за исключением случаев из таблицы словосочетаний)
                            supress_accent = False

                            if word in no_accent_words:
                                supress_accent = True
                            elif ud_token.upos in ('PRON', 'ADV', 'DET') and nvowels == 1:
                                # Односложные наречия, местоимения и т.д.
                                is_optional_stress = False  #True
                            elif ud_token.upos == 'ADP' and count_vowels(word) <= 1:
                                supress_accent = True
                                is_optional_stress = False
                            elif ud_token.upos == 'PART' and word in ['и', 'а', 'же', 'ка', 'то', 'ли', 'бы', 'уж',
                                                                      'не', 'но', 'ни', 'аж']:
                                supress_accent = True
                                is_optional_stress = False
                            elif ud_token.upos in ('PUNCT', 'ADP', 'PART', 'SCONJ', 'CCONJ', 'INTJ'):
                                supress_accent = True
                                is_optional_stress = False  #True
                            elif word in opt_words:
                                is_optional_stress = True

                            if supress_accent:
                                stress_pos = -1
                            elif word in self.accentuator.ambiguous_accents2:
                                # для слов типа пОнял-понЯл, где есть варианты ударения для одной формы
                                # Подразумевается, что в словаре ambiguous_accents2.json первый вариант ударения - основной.
                                item = TokenStressVariants()
                                for stress_pos in self.accentuator.ambiguous_accents2[word]:
                                    if stress_pos == -1:
                                        raise RuntimeError()
                                    g = TokenGroup()

                                    pword = PoetryWord(ud_token.lemma, ud_token.form, ud_token.upos, ud_token.tags, [WordAccentuation(stress_pos, None)])
                                    g.add(pword)
                                    item.add(g)

                                res_items.append(item)
                                iword += 1
                                continue
                            elif not is_optional_stress:
                                stress_pos = self.accentuator.get_accent(word, ud_tags=ud_token.tags + [ud_token.upos])
                            else:
                                if random.random() > 0.5:
                                    stress_pos = self.accentuator.get_accent(word, ud_tags=ud_token.tags + [ud_token.upos])
                                else:
                                    stress_pos = -1

                            # Одиночное слово.
                            pword = PoetryWord(ud_token.lemma, ud_token.form, ud_token.upos, ud_token.tags, [WordAccentuation(stress_pos, None)])

                            g = TokenGroup()
                            g.add(pword)

                            item = TokenStressVariants()
                            item.add(g)

                            res_items.append(item)

                            iword += 1

                    stressed_lines.append(RapLine(res_items))

                # Все строки в блоке размечены, результаты разметки собраны в stressed_lines
                line_variants = [line.get_rhyming_variants(self) for line in stressed_lines]

                # Теперь надо выбрать те варианты ударения для каждой строки, чтобы максимизировать общую рифмовку.
                rhyming_graf = []
                iline = 0
                while iline < len(line_variants):
                    # Попробуем подобрать вариант сразу для 4х строк полным перебором вариантов.
                    if len(line_variants) - iline >= 4:
                        lines4 = line_variants[iline: iline+4]

                        best_variants4 = None
                        best_rhyming = None
                        best_num_rhymed = 0

                        for variant4 in itertools.product(*lines4):
                            r12 = self.check_rhyming(variant4[0].rhyming_tail, variant4[1].rhyming_tail)
                            r13 = self.check_rhyming(variant4[0].rhyming_tail, variant4[2].rhyming_tail)
                            r14 = self.check_rhyming(variant4[0].rhyming_tail, variant4[3].rhyming_tail)

                            r23 = self.check_rhyming(variant4[1].rhyming_tail, variant4[2].rhyming_tail)
                            r24 = self.check_rhyming(variant4[1].rhyming_tail, variant4[3].rhyming_tail)

                            r34 = self.check_rhyming(variant4[2].rhyming_tail, variant4[3].rhyming_tail)

                            rhyming = tuple()

                            if r12 and r34 and not r23:
                                # AABB
                                rhyming = (1, 0, 1, 0)
                            elif r13 and r24 and not r12 and not r34:
                                # ABAB
                                rhyming = (2, 2, 0, 0)
                            elif r14 and r23 and not r12:
                                # ABBA
                                rhyming = (3, 1, 0, 0)
                            elif r12 and r23 and r34:
                                # AAAA
                                rhyming = (1, 1, 1, 0)

                            if rhyming:
                                num_rhymed = sum(z!=0 for z in rhyming)
                                if num_rhymed > best_num_rhymed:
                                    best_num_rhymed = num_rhymed
                                    best_variants4 = list(variant4)
                                    best_rhyming = rhyming

                        if best_variants4:
                            line_variants[iline: iline + 4] = [[variant] for variant in best_variants4]
                            rhyming_graf.extend(best_rhyming)
                            iline += 4
                            continue

                    # Упрощенный алгоритм выбора вариантов для оставшихся строк.
                    while iline < len(line_variants):
                        if iline == len(line_variants)-1:
                            rhyming_graf.append(0)
                            iline += 1
                        else:
                            best_variant_1 = None
                            best_variant_2 = None
                            best_edge = 0

                            for edge in [1,2,3]:
                                iline2 = iline + edge
                                if iline2 < len(line_variants):
                                    for variant_1 in line_variants[iline]:
                                        for variant_2 in line_variants[iline2]:
                                            rij = self.check_rhyming(variant_1.rhyming_tail, variant_2.rhyming_tail)
                                            if rij:
                                                best_edge = edge
                                                best_variant_1 = variant_1
                                                best_variant_2 = variant_2
                                                break

                            if best_edge:
                                rhyming_graf.append(best_edge)
                                line_variants[iline] = [best_variant_1]
                                line_variants[iline + best_edge] = [best_variant_2]
                                iline += 1  #best_edge
                            else:
                                rhyming_graf.append(0)
                                iline += 1

                # Оставляем первый вариант ударения для тех строк, по которым выбор не получилось сделать через рифмовку.
                stressed_lines = [vx[0] for vx in line_variants]

                score = sum(g!=0 for g in rhyming_graf) / (len(line_variants) * 0.5 + 1e-6)

                alignment = RapBlockAlignment(block_header, block_lines, stressed_lines, rhyming_graf, score)
                aligned_blocks.append(alignment)

        alignment = RapAlignment(rap_text, aligned_blocks)
        return alignment


class RapBlockAlignment(object):
    def __init__(self, block_header, block_lines, stressed_lines, rhyming_graf, score):
        self.block_header = block_header
        self.block_lines = list(block_lines)
        self.stressed_lines = list(stressed_lines)
        self.rhyming_graf = list(rhyming_graf)
        self.score = score

    def __repr__(self):
        chunks = []
        if self.block_header:
            chunks.append(self.block_header)
        if self.stressed_lines:
            chunks.extend(map(str, self.stressed_lines))
        return '\n'.join(chunks)

    def get_stressed_lines(self):
        text_representation = ''
        if self.block_header:
            text_representation = self.block_header

        if self.stressed_lines:
            text_representation = text_representation + '\n' + '\n'.join(line.get_stressed_line() for line in self.stressed_lines)

        return text_representation


class RapAlignment(object):
    def __init__(self, text, blocks):
        self.text = text
        self.blocks = list(blocks)
        self.total_score = np.mean([b.score for b in blocks])

    def get_total_score(self) -> float:
        return self.total_score

    def __repr__(self):
        chunks = []
        if self.blocks:
            chunks = list(map(str, self.blocks))
        return '\n\n'.join(chunks)

    def get_rhyming_graph(self):
        return list(itertools.chain(*[block.rhyming_graf for block in self.blocks]))

    def get_stressed_lines(self) -> str:
        text = '\n\n'.join(block.get_stressed_lines() for block in self.blocks)
        text = normalize_whitespaces(text)
        return text

    def get_markup(self) -> str:
        return self.get_stressed_lines()


class StressedToken(object):
    def __init__(self, form, stress_pos):
        self.form = form
        self.stress_pos = stress_pos
        if stress_pos != -1:
            self.stressed_form = render_word_with_stress(form, stress_pos)
        else:
            self.stressed_form = form

    def __repr__(self):
        return self.stressed_form

class TokenGroup(object):
    def __init__(self):
        self.pwords = []

    def add(self, pword):
        self.pwords.append(pword)

    def __repr__(self):
        return ' '.join(map(str, self.pwords))


class TokenStressVariants(object):
    def __init__(self):
        self.variants = []

    def add(self, group: TokenGroup):
        self.variants.append(group)

    def __repr__(self):
        return '|'.join(map(str, self.variants))


class RapLine(object):
    def __init__(self, items: List[TokenStressVariants]):
        self.items = list(items)

    def __repr__(self):
        return '  '.join(map(str, self.items))

    def get_rhyming_variants(self, aligner):
        variants = []

        head_pwords = []
        for item in self.items[:-1]:
            item_variant1 = item.variants[0]
            for pword in item_variant1.pwords:
                head_pwords.append(pword)

        for item in self.items[-1].variants:
            tail_pwords = []
            for pword in item.pwords:
                tail_pwords.append(pword)

            variant_swords = [pword.get_first_stress_variant(aligner) for pword in (head_pwords + tail_pwords)]

            variants.append(RapLineVariant(variant_swords))

        return variants


class RapLineVariant(object):
    def __init__(self, pwords):
        self.stressed_words = pwords
        self.rhyming_tail = None
        self.init_rhyming_tail()

    def __repr__(self):
        return '  '.join(map(str, self.stressed_words))

    def get_stressed_line(self) -> str:
        return ' '.join(map(str, self.stressed_words))

    def init_rhyming_tail(self):
        stressed_word = None
        unstressed_prefix = None
        unstressed_postfix_words = []

        # Ищем справа слово с ударением
        i = len(self.stressed_words)-1
        while i >= 0:
            pword = self.stressed_words[i]
            if pword.new_stress_pos != -1:  # or pword.poetry_word.n_vowels > 1:
                stressed_word = pword

                if re.match(r'^[аеёиоуыэюя]$', pword.form, flags=re.I) is not None:
                    # Ситуация, когда рифмуется однобуквенное слово, состоящее из гласной:
                    #
                    # хочу́ отшлё́пать анако́нду
                    # но непоня́тно по чему́
                    # вот у слона́ гора́здо ши́ре
                    # чем у                       <=======
                    if i > 0:
                        unstressed_prefix = self.stressed_words[i-1].poetry_word.form[-1].lower()

                # все слова, кроме пунктуации, справа от данного сформируют безударный хвост клаузуллы
                for i2 in range(i+1, len(self.stressed_words)):
                    if self.stressed_words[i2].poetry_word.upos != 'PUNCT':
                        unstressed_postfix_words.append(self.stressed_words[i2])

                break
            i -= 1

        self.rhyming_tail = RhymingTail(unstressed_prefix, stressed_word, unstressed_postfix_words)

    def count_syllables(self) -> int:
        return sum(word.count_syllables() for word in self.stressed_words)


class SongBlock(object):
    def __init__(self, header, lines):
        self.block_header = header
        self.block_lines = list(lines)


def render_word_with_stress(word, stress_pos):
    cx = []
    vowel_counter = 0
    for c in word:
        cx.append(c)
        if c.lower() in 'уеыаоэёяию':
            vowel_counter += 1
            if vowel_counter == stress_pos:
                cx.append('\u0301')
    token2 = ''.join(cx)
    return token2


def is_depresyashka_rhyming(alignment) -> bool:
    if alignment.rhyme_scheme in ('-A-A', 'ABAB'):
        return True
    else:
        if len(alignment.poetry_lines) != 4:
            return False

        # Обработаем случаи, когда в последней строке депрессяшки безударное окончание рифмуется:
        #
        # ми́лая ули́тка
        # вы́суни рога́
        # забода́й родна́я
        # ты́ нарко́лога
        #
        sx2 = alignment.poetry_lines[1].split_to_syllables()
        sx4 = alignment.poetry_lines[3].split_to_syllables()
        if sx2[-1].replace('\u0301', '') == sx4[-1]:
            #if alignment.poetry_line[3].stress_signature_str == '10100':
            return True

        # все́ клие́нты пря́чут
        # в ма́ске по́л лица́
        # бу́дто ба́нк огра́бить
        # собира́юцца
        tail2 = alignment.poetry_lines[3].get_rhyming_tail().get_text().replace('\u0301', '')
        if tail2.endswith(sx2[-1].replace('\u0301', '')):
            return True

        return False

def is_poroshok_rhyming(alignment) -> bool:
    if alignment.rhyme_scheme in ('-A-A', 'ABAB'):
        return True
    else:
        # Обработаем случаи, когда в последней строке порошка безударное окончание рифмуется:
        #
        # почу́яв за́пахи мажо́ров
        # бегу́ лома́я каблуки́
        # но не впусти́ли в клу́б эли́тный
        # су́ки
        sx2 = alignment.poetry_lines[1].split_to_syllables()
        sx4 = alignment.poetry_lines[3].split_to_syllables()
        if sx2[-1].replace('\u0301', '') == sx4[-1].replace('\u0301', ''):
            return True

        tail2 = alignment.poetry_lines[3].get_rhyming_tail().get_text().replace('\u0301', '')
        if tail2.endswith(sx2[-1].replace('\u0301', '')):
            return True

        return False


if __name__ == '__main__':
    proj_dir = os.path.expanduser('~/polygon/text_generator')
    tmp_dir = os.path.join(proj_dir, 'tmp')
    data_dir = os.path.join(proj_dir, 'data')
    models_dir = os.path.join(proj_dir, 'models')

    output_dir = os.path.join(models_dir, 'scansion_tool')
    pathlib.Path(output_dir).mkdir(exist_ok=True)

    PoetryStressAligner.compile(data_dir, output_dir)

    # collocations = collections.defaultdict(list)
    #
    # with io.open(os.path.join(data_dir, 'poetry/dict/collocation_accents.dat'), 'r', encoding='utf-8') as rdr:
    #     for line in rdr:
    #         line = line.strip()
    #         if line.startswith('#') or len(line) == 0:
    #             continue
    #
    #         c = CollocationStress.load_collocation(line)
    #
    #         collocations[c.key()].append(c)
    #
    # word_segmentation = dict()
    # with io.open(os.path.join(data_dir, 'poetry/dict/word_segmentation.csv'), 'r', encoding='utf-8') as rdr:
    #     header = rdr.readline()
    #     for line in rdr:
    #         fields = line.strip().split(',')
    #         word = fields[0]
    #         segments = fields[1].split('/')
    #         word_segmentation[word] = WordSegmentation(segments)
    #
    # with open(os.path.join(output_dir, 'scansion_tool.pkl'), 'wb') as f:
    #     pickle.dump(collocations, f)
    #     pickle.dump(word_segmentation, f)

    print('All done :)')
