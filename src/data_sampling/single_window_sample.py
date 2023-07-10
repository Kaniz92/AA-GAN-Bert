def middle_sample(text, max_seq_length=512, sample_size = 1):
    input_text = str(text).split()
    book_text = " ".join(input_text)

    del input_text

    mid_position = len(book_text.split()) / 2
    # here we get text phrase from the middle to avoid table of content or any footers

    if mid_position + 100 > max_seq_length:
        start = int(mid_position - max_seq_length / 2)
        end = int(mid_position + max_seq_length / 2)
        result_list = book_text.split()[start:end]
        result_str = " ".join(result_list)

        return result_str

def n_samples_sequential(text, max_seq_length=512, sample_size=1):
    input_text = str(text).split()
    book_text = " ".join(input_text)

    del input_text

    sample_list = []    
    mid_position = 0
    for i in range(int(sample_size)):
      mid_position = i+max_seq_length / 2+1 +mid_position

      if mid_position+max_seq_length / 2 < len(book_text.split()):
        if mid_position>max_seq_length / 2:
          start = int(mid_position - max_seq_length / 2)
          end = int(mid_position + max_seq_length / 2)
          result_list = book_text.split()[start:end]
          result_str = " ".join(result_list)
          sample_list.append(result_str)

    return sample_list
