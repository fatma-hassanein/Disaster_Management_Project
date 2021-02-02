import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    '''
    INPUT 
        messages_filepath: Name of file that contains the messages data
        categories_filepath: Name of the file that contains the categories data
    OUTPUT
        Returns a dataframe after loading both datasets from their file and joining them 
        on the common key 'id'
    '''
        
    # load datasets
    messages = pd.read_csv(messages_filepath)
    categories =  pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, how='outer',on=['id', 'id'])
    return df

def clean_data(df):
    
    '''
    INPUT 
        df: Loaded data from files  
    OUTPUT
        Returns the df after cleaning the categories data by:
        - Splitting them into 36 columns
        - Changing their values to binary (0 or 1)
        - Dropping any duplicate rows
        - Rejoining the cleaned columns to the original df and removing categories column
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";",expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.str.split("-",expand=True)
    category_colnames = category_colnames[0]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    new_categories = categories.copy()
    for column in new_categories:
        # set each value to be the last character of the string
        new_categories[column] = new_categories[column].astype(str).str.split("-",expand=True)[1]
        # convert column from string to numeric
        new_categories[column] = pd.to_numeric(new_categories[column])
        new_categories[column][new_categories[column] >= 1] = 1
        new_categories[column][new_categories[column] < 1] = 0
     
    # drop the original categories column from `df`
    del df['categories']
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,new_categories], axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    
    '''
    INPUT 
        df: Final Cleaned dataframe
        database_filename: Name of Database to save the cleaned data  
    OUTPUT
        Saves the final cleaned dataframe to a table in the database
    '''
    # Create engine
    engine = create_engine('sqlite:///././'+database_filename)

    # Add cleaned data to new table
    df.to_sql('Messages_with_Cats', engine, index=False, if_exists='replace')


def main():
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()