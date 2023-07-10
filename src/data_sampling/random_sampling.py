import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


# function that creates random sample
# case_step = gap between cases
def random_sampling(population, case_step, random_sample_size, path, author_info):
    cases_df = pd.DataFrame()

    for case_id in range(2, len(population), case_step):
        author_ids = []
        author_names = []
        case_ids = []

        # generate additional 50 in case of duplicates
        for set_id in range(1, random_sample_size + 50):
            random_sample_ids = np.random.choice(population, replace=False, size=case_id)

            # remove any duplicates generated
            if not any(np.array_equal(random_sample_ids, elem) for elem in author_ids):
                # author_ids.append(random_sample_ids)
                author_id_list = []

                random_sample_names = []
                for i in random_sample_ids:
                    random_sample_names.append(list(author_info.values())[int(i)])
                    author_id_list.append(list(author_info.keys())[int(i)])
                random_sample_names = ','.join(random_sample_names)
                author_names.append(random_sample_names)
                author_ids.append(author_id_list)

                case_ids.append(case_id)

        sets_df = pd.DataFrame()

        sets_df['case_id'] = case_ids
        sets_df['author_ids'] = author_ids
        sets_df['author_names'] = author_names

        # get only the required count
        sets_df = sets_df[:random_sample_size]
        sets_df.insert(1, 'set_id', np.arange(1, random_sample_size + 1))

        cases_df = cases_df.append(sets_df)

    cases_df.to_csv(f'{path}/random_sampling_{random_sample_size}.csv', index=False)


def generate_random_samples(random_sample_size, path):
    author_info = {
        324: 'Anthony Trollope',
        69: 'Arthur Conan Doyle',
        593: 'Bret Harte',
        1057: 'Fergus Hume',
        73: 'Frances Hodgson Burnett',
        30: 'H.G. Wells',
        365: 'Henry Rider Haggard',
        120: 'Jack London',
        8949: 'James Grant',
        979: 'John Kendrick Bangs',
        125: 'Joseph Conrad',
        102: 'Louisa May Alcott',
        963: 'Margaret Oliphant',
        1321: 'Marie Corelli',
        53: 'Mark Twain',
        2957: 'Mary Elizabeth Braddon',
        1177: 'Mrs Henry Wood',
        28: 'Nathaniel Hawthorne',
        260: 'Oliver Optic',
        98: 'Wilkie Collins'
    }

    population = np.arange(0, len(author_info))
    case_step = 2
    random_sampling(population, case_step, random_sample_size, path, author_info)


def get_dataset_split_per_row(valid_df_selected, row, sample_size=20):
    author_list = row['author_names'].split(',')
    book_ids = []

    for name in author_list:
        book_ids = book_ids + valid_df_selected[valid_df_selected['Name'] == name].head(int(sample_size))[
            'BookID'].tolist()

    book_list = valid_df_selected[valid_df_selected['BookID'].isin(book_ids)]

    trainval, test = train_test_split(book_list,
                                      test_size=0.2,
                                      stratify=book_list['AuthorID'])
    train, val = train_test_split(trainval, test_size=0.1, stratify=trainval['AuthorID'])

    return list(train['BookID']), list(test['BookID']), list(val['BookID'])


def generate_dataset_split_indexes(random_sample_size, path, huggingface_repo):
    data = pd.read_csv(f'{path}/random_sampling_{random_sample_size}.csv')
    main_dataset = load_dataset("Authorship/master-data", data_files="main_data.csv")

    valid_df_selected = pd.DataFrame(main_dataset['train'])
    valid_df_selected = valid_df_selected[valid_df_selected['Genre'] == 'Novel']

    data['train'], data['test'], data['val'] = zip(
        *data.apply(lambda x: get_dataset_split_per_row(valid_df_selected, x), axis=1))

    # save locally
    data.to_csv(f'{path}/random_sampling_{random_sample_size}_split_index.csv', index=False)

    # save in huggingface
    # uncomment when huggingface dataset repo created
    # data.to_csv(f'{huggingface_repo}/indexes.csv', index=False)
