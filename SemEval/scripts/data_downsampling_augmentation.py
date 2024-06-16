# This script takes a pre-processed training dataset and balance out the datapoints in each label by performing downsampling or data augmentation by back-translation.

from easynmt import EasyNMT
import pandas as pd
import random
import argparse
import os

def back_translate(texts,model,intermediate_lang):
  """
  Back translate a list of provided texts through an intermediate language to generate synthetic datapoints for data augmentation

  Args:
      texts (list of str): List of texts to be back-translated
      model (EasyNMT): The translation model
      intermediate_lang (str): Language code of the intermediate language, e.g. 'es', 'fr'

  Returns:
      back_translated (list of str): Back-translated text
  """
  translated = model.translate(texts, source_lang='en', target_lang=intermediate_lang)
  back_translated = model.translate(translated, source_lang=intermediate_lang, target_lang='en')

  return back_translated

def main():
  """
  Wrapper function to execute the script.
  """
  translation_model = EasyNMT('opus-mt')

  parser = argparse.ArgumentParser()
  parser.add_argument("--train_df_file",'-t',type=str,required=True,help="Paths to the training dataset with processed_text column.")
  parser.add_argument("--n_sample_to_achieve",'-n',type=int,required=True,help="Number of samples to downsample/augment to achieve")
  args = parser.parse_args()
  
  train_df_file = args.train_df_file
  n_sample = args.n_sample_to_achieve

  train_df = pd.read_csv(train_df_file,index_col=0)
  balanced_train_df = pd.DataFrame(columns=train_df.columns)
  balanced_train_df.index.names = train_df.index.names
  balanced_train_df = balanced_train_df.astype(train_df.dtypes.to_dict())

  output_file = f'{os.path.split(train_df_file)[0]}/{n_sample}_balanced_train_df.csv'

  print(f'Begin to downsample/augment to achieve {n_sample} samples in all labels!')

  labels = ['Appeal to authority', 'Appeal to fear/prejudice', 'Bandwagon',
          'Black-and-white Fallacy/Dictatorship', 'Causal Oversimplification',
          'Doubt', 'Exaggeration/Minimisation', 'Flag-waving',
          'Glittering generalities (Virtue)', 'Loaded Language',
          "Misrepresentation of Someone's Position (Straw Man)",
          'Name calling/Labeling',
          'Obfuscation, Intentional vagueness, Confusion',
          'Presenting Irrelevant Data (Red Herring)', 'Reductio ad hitlerum',
          'Repetition', 'Slogans', 'Smears', 'Thought-terminating cliché',
          'Whataboutism']
  asc_sorted_label_dict = dict(train_df[labels].sum().sort_values())

  # start by augmenting data in the labels with insufficient (the lowest) data.
  # This is because the original data might have other labels annotated as 1 too, and they will be appended into the balanced_train_df, 
  # to avoid the texts with many labels annotated as 1 from imbalancing the data again after subsampling, I will iteratively check how many data are needed for 
  # augmentation or subsampling, and doing so for only that amount
  for label in asc_sorted_label_dict:
    print(f'Processing {label}!')

    # calculate how many data is needed for a given label
    n_current = balanced_train_df[balanced_train_df[label] == 1].shape[0]
    n_sample_to_generate = n_sample - n_current

    if n_sample_to_generate > 0: # append more data to balanced_train_df either by back-translation or subsampling from the train_df
      selected_df = train_df[train_df[label] == 1].copy()
      texts = selected_df['processed_text'].to_list()

      if selected_df.shape[0] >= n_sample_to_generate: # enough data to just randomly sample and append to balanced_train_df
        balanced_train_df = pd.concat([balanced_train_df,selected_df.sample(n=n_sample_to_generate,replace=False)],axis=0)

      else: # augment by back_translation
        balanced_train_df = pd.concat([balanced_train_df,selected_df],axis=0) # añadir los textos originales

        # re-calculate how many data is needed for a given label after appending the original data from train_df to balanced_train_df
        n_current = balanced_train_df[balanced_train_df[label] == 1].shape[0]
        n_sample_to_generate = n_sample - n_current

        if n_sample_to_generate <= 0:
          print(f'{label}: {n_current} achieved after appending from train_df, no balancing performed on this label')
        else:
          print(f'{label}: augmenting {n_sample_to_generate} datapoints to achieve {n_sample} datapoints')
          if len(texts) >= n_sample_to_generate: 
            print(f'More data ({len(texts)}) available for data augmentation than needed, randomly sampling {n_sample_to_generate} for back-translation!')
            texts = random.sample(texts,k=n_sample_to_generate)
            augment_texts = back_translate(texts,model=translation_model,intermediate_lang='es')
          else: # if less texts is available than needed, randomly sample one and translate and repeat until enough back-translated text is obtained
            print(f'Fewer data ({len(texts)}) available for data augmentation, randomly sampling available data for back-translation!')
            augment_texts = []
            while n_sample_to_generate > len(augment_texts):
              augment_texts.extend(back_translate(random.sample(texts,k=1),model=translation_model,intermediate_lang='es'))

          # construct the df for pd.concat
          augment_df = pd.DataFrame(columns=train_df.columns)
          augment_df['processed_text'] = augment_texts
          augment_df[label] = 1 
          augment_df['text'] = 'dummy' # para que sea un string y es consistente con train_df
          augment_df['link'] = 'dummy' # para que sea un string y es consistente con train_df
          # IMPORTANT: Since I observed that correlations between labels are weak, I just fill the label intended for augmentation as 1 and leave the rest as 0, 
          # it could very well be that the data has other labels annotated with 1 too, but these cases are ignored
          augment_df.fillna(0,axis=0,inplace=True)
          # create random indices that do not coincide with the indices from train_df and balanced_train_df to use as the indices for balanced_train_df
          rand_ind = []
          while len(rand_ind) < augment_df.shape[0]:
            randint = random.randint(a=1,b=999999)
            if randint not in train_df.index.to_list() + balanced_train_df.index.to_list():
              rand_ind.append(randint)
          augment_df.index = rand_ind
          balanced_train_df = pd.concat([balanced_train_df,augment_df],axis=0)
    balanced_train_df.index.names = train_df.index.names
    balanced_train_df = balanced_train_df.astype(train_df.dtypes.to_dict())

    balanced_train_df.to_csv(output_file)
    print(f'Done processing {label} and saved as {output_file}!')

  print('Final balanced train df:\n')
  print(print(balanced_train_df[labels].sum()))

if __name__ == '__main__':
  main()

# python3 data_downsampling_augmentation.py -t ./train_df.csv -n 600