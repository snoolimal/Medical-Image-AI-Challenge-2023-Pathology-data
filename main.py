from preprocessing.removing_background import BGRemover


def main():
    BGREMOVER = BGRemover()
    BGREMOVER.remove_background('train')
    BGREMOVER.remove_background('test')


if __name__ == '__main__':
    main()
