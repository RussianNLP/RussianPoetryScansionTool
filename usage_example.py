import os
import terminaltables

from phonetic import Accents
from poetry_alignment import PoetryStressAligner
from udpipe_parser import UdpipeParser


if __name__ == '__main__':
    proj_dir = os.path.expanduser('.')
    models_dir = os.path.join(proj_dir, 'models')

    parser = UdpipeParser()
    parser.load(models_dir)

    accents = Accents(device="cpu")
    accents.load_pretrained(os.path.join(models_dir, 'accentuator'))

    aligner = PoetryStressAligner(parser, accents, model_dir=os.path.join(models_dir, 'scansion_tool'))
    aligner.max_words_per_line = 14

    poem = """Вменяйте ж мне в вину, что я столь мал,
Чтоб за благодеянья Вам воздать,
Что к Вашей я любви не воззывал,
Чтоб узами прочней с собой связать,
Что часто тёмным помыслом я сам
Часы, Вам дорогие столь, дарил,
Что я вверялся часто парусам,
Чей ветр меня от Вас вдаль уносил.
Внесите в список Ваш: мой дикий нрав,
Ошибки, факты, подозрений ложь,
Но, полностью вину мою признав,
Возненавидя, не казните всё ж."""

    a = aligner.align(poem.strip().split('\n'))

    print('score={} meter={} scheme={}'.format(a.score, a.meter, a.rhyme_scheme))
    print(a.get_stressed_lines(show_secondary_accentuation=True))

    print('\n')
    table = [['Stress pattern', 'Accentuation', 'Verse']]
    for line, pline, mapping in zip(a.get_stressed_lines(show_secondary_accentuation=True).split('\n'), a.poetry_lines, a.metre_mappings):
        table.append((mapping.get_stress_signature_str(), mapping, line))
    print(terminaltables.AsciiTable(table).table)
