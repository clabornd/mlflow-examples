import polars as pl
from sklearn.preprocessing import TargetEncoder
import yaml
import os

class PostgresDataSet:
    """
    A class to represent a dataset from a PostgreSQL database using Polars.
    
    Attributes:
        config_path (str): Path to the YAML configuration file containing PostgreSQL connection details.
        query (str): SQL query to fetch data from the PostgreSQL database.
        raw (polars.DataFrame): The raw data fetched from the PostgreSQL database.
    
    Methods:
        preprocess_X(df, **kwargs):
            Preprocesses the feature data.
        preprocess_y(df, **kwargs):
            Preprocesses the target data.
        get_data():
            Returns the raw data fetched from the PostgreSQL database.
    """
    def __init__(self, config_path=None, query=None):
        """
        Constructs all the necessary attributes for the PostgresDataSet object.
        
        Args:
            config_path (str, optional): Path to the YAML configuration file. Defaults to None.
            query (str, optional): SQL query to fetch data. Defaults to None.
        """
        self.config_path = config_path 
        self.query = query
    
        connstr = "Driver={PostgreSQL};Server={postgres_host};Port={postgres_port};Database={postgres_db};Uid={postgres_user};Pwd={postgres_password}"
        
        if self.config_path is not None:
            postgres_config = yaml.load(open(self.config_path, "r"), Loader=yaml.FullLoader)
            postgres_db = postgres_config['POSTGRES_DB']
            postgres_port = postgres_config['POSTGRES_PORT']
            postgres_host= postgres_config['POSTGRES_HOST']
            postgres_user = postgres_config['POSTGRES_USER']
            postgres_password = postgres_config['POSTGRES_PASSWORD']
        else:
            postgres_db = os.environ['POSTGRES_DB']
            postgres_port = os.environ['POSTGRES_PORT']
            postgres_host = os.environ['POSTGRES_HOST']
            postgres_user = os.environ['POSTGRES_USER']
            postgres_password = os.environ['POSTGRES_PASSWORD']

        connstr = connstr.format(
            PostgreSQL = "{PostgreSQL}",
            postgres_db = postgres_db,
            postgres_port = postgres_port,
            postgres_host = postgres_host,
            postgres_user = postgres_user,
            postgres_password = postgres_password
        )

        df = pl.read_database(
            connection = connstr,
            query = self.query or "SELECT * FROM athlete_events;"
        )

        self.raw = df
   
    def preprocess_X(self, df, **kwargs):
        return(df)

    def preprocess_y(self, df, **kwargs):
        return(df)
    
    def get_data(self):
        return self.raw

class OlympicsDataset(PostgresDataSet):
    """
    A dataset class for handling the Olympics dataset stored in a PostgreSQL database.

    Attributes:
        encoders (dict): A dictionary to store encoders for categorical columns.

    Methods:
        impute_X(df):
            Imputes missing values in the DataFrame by filling them with the median value.
        preprocess_X(df, **kwargs):
            Preprocesses the DataFrame by encoding categorical columns and imputing missing values.
        encode_struct(struct, variable, target, encoder, do_fit=True):
            Encodes a structured column using a target encoder.
        get_train_test_splits():
            Splits the raw data into training and testing sets, preprocesses them, and returns the splits.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoders = {}

    def impute_X(self, df):
        df = df.with_columns(pl.all().fill_null(pl.all().median()))

        return df

    def preprocess_X(self, df, **kwargs):
        cat_cols = [col for col, t in zip(df.columns, df.dtypes) if t == pl.String and col != "medal"]

        for cc in cat_cols:
            enc = self.encoders.get(cc) or TargetEncoder()
            do_fit = self.encoders.get(cc) is None

            df = df.with_columns(
                pl.struct([cc, 'medal']).map_batches(
                    lambda x: self.encode_struct(x, cc, 'medal', enc, do_fit)
                ).alias(f"{cc}_encoding")
            ).unnest(f"{cc}_encoding")

            self.encoders[cc] = enc

        X_train = X_train.with_columns(pl.all().fill_null(pl.all().median()))

        # drop the encoded categorical columns
        df = df.drop(cat_cols + ['index', 'id'])

        return self.impute_X(df)

    def encode_struct(self, struct, variable, target, encoder, do_fit=True):
        X = struct.struct.field(variable).to_numpy()
        y = struct.struct.field(target).to_numpy()

        if do_fit:
            embed = encoder.fit_transform(X[:,None], y)
        else:
            embed = encoder.transform(X[:,None])

        # convert it to a polars dataframe with one column per
        embed = pl.DataFrame(embed)
        embed = embed.rename(
            {col:f"{variable}_{cls}" for col, cls in zip(embed.columns, encoder.classes_)}
        )

        out_struct = embed.select(
            pl.struct(embed.columns).alias(f"{variable}_struct")
        )

        return out_struct.to_series()
    
    def get_train_test_splits(self):
        df = self.raw.with_columns(pl.col("medal").fill_null('DNP'))

        train_idx = (
            df.with_row_index("row_idx")
            .group_by("medal")
            .agg([pl.col("row_idx").sample(fraction = 0.8, with_replacement=False, seed=43434)])
            .explode("row_idx")
        )

        X_train = (
            df.with_row_index()
            .filter(pl.col('index').is_in(train_idx['row_idx']))
        )
        X_train = self.preprocess_X(X_train) # sets the encoders
        y_train = X_train.select(pl.col("medal"))
        X_train = X_train.drop("medal")

        X_test = (
            df.with_row_index()
            .filter(~pl.col('index').is_in(train_idx['row_idx']))
        )
        X_test = self.preprocess_X(X_test)
        y_test = X_test.select(pl.col("medal"))
        X_test = X_test.drop("medal")

        return X_train, X_test, y_train, y_test