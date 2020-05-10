import random
import wikipedia


def getData(language):
    wikipedia.set_lang(language)

    pages = wikipedia.random(2000)
    print(pages)
    content = []
    for page in pages:
        try:
            summary = wikipedia.summary(page)
            summary = summary.split()

            for i in range(0, len(summary), 15):
                content.append(summary[i: i+15])
        except:
            print("inside error block")
    return content


def main():
    language = ['en', 'nl']
    data = []
    for lang in language:
        print(lang)
        temp = getData(lang)
        for t in temp:
            if len(t) == 15:
                data.append(lang + "|" + " ".join(t))

    print(data)
    random.shuffle(data)
    file = open("testing_data.txt", "w", encoding='utf8')
    for d in data:
        file.write(d)
        file.write("\n")


if __name__ == '__main__':
    main()