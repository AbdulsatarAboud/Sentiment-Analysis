from classes.vader.SentimentIntensityAnalyzer import SentimentIntensityAnalyzer
from constants.general import DATA_DIR
import pandas

if __name__ == '__main__':

    df = pandas.read_csv(DATA_DIR+"test.txt", sep="|")

    analyzer = SentimentIntensityAnalyzer()

    print("-----------------------START-----------------------------")

    for sentence in df['review']:
        vs = analyzer.polarity_scores(sentence)
        print("{:-<65} {}".format(sentence, str(vs)))

    print("----------------------THE END------------------------------")
