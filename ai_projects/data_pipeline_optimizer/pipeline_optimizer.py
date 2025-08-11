import random
import numpy as np
import pandas as pd 
from copy import deepcopy
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, make_scorer
import warnings
warnings.filterwarnings("ignore")

#Find the best sequence of data preprocessing steps that improves classification model performance on a synthetic dataset.

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

#1. Synthetic dirty data
def generate_dirty_dataset(n_samples=1000, n_numric=6, n_categorical=2, missing_frac=0.12, outlier_frac=0.02):
    x_num, y = make_classification(n_samples=n_samples,
                                   n_features=n_numric,
                                   n_informative=3,
                                   n_redundant=1,
                                   random_state=random_seed)
    df = pd.DataFrame(x_num, columns = [f"num_{i}" for i in range(n_numric)])

    #Add categorical columns
    for i in range(n_categorical):
        #Make categories influenced by target
        cats = np.where(y==1,
                        np.random.choice(["A","B","C"], size=n_samples, p=[0.6,0.3,0.1]),
                        np.random.choice(["A","B","C"], size=n_samples, p=[0.3,0.6,0.1]))
        df[f"cat_{i}"] = cats
    
    #Introduce missing values
    for col in df.columns:
        mask = np.random.rand(n_samples) < missing_frac
        df.loc[mask,col] = np.nan

    #Add some outliers to numric columns 
    for col in [c for c in df.columns if c.startswith("num_")]:
        mask = np.random.rand(n_samples) < outlier_frac
        df.loc[mask,col] = df[col].mean() + 10 * df[col].std() * np.random.randn(mask.sum())

    #Introduce inconsistent categorical strings/noise
    df['cat_0'] = df['cat_0'].astype(object)
    noise_mask = np.random.rand(n_samples) < 0.2
    df.loc[noise_mask, 'cat_0'] = 'Unknown/' + df.loc[noise_mask, 'cat_0'].astype(str)

    #Add a text-like date column messy format
    dates = pd.date_range("2020-01-01", periods=n_samples, freq='D')
    dates = dates.astype(str) 

    #Randomly change some formarts
    for i in range(0, n_samples, 50):
        dates[i] = dates[i].replace('-','/')
    df['date_messy'] = dates

    #Random deletion
    df.loc[np.random.rand(n_samples) < 0.5, 'date_messy'] = np.nan

    df['target'] = y
    return df 


#2. Transformers - Data cleaning
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        return x[self.cols]

class OutlierRemover(BaseEstimator, TransformerMixin):
    """Simple IQR based outlier removel (replace with nan, so next imputer handles)
    mode: iqr or zscore
    z_thresh: threshold for zscore"""
    def __init__(self, method='iqr'):
        self.method = method
    
    def fit(self, x, y=None):
        return self 
    
    def transform(self, x):
        x = x.copy()
        num_cols = [c for c in x.columns if x[c].dtype.kind in 'fi']
        if self.method == 'iqr':
            for c in num_cols:
                q1 = x[c].quantile(0.25)
                q3 = x[c].quantile(0.75)
                iqr = q3 - q1 
                low = q1 - 1.5 * iqr
                high = q3 + 1.5 * iqr 
                mask = (x[c] < low) | (x[c] > high)
                x.loc[mask, c] = np.nan
        else:
            #Zscore
            for c in num_cols:
                m = x[c].mean()
                s = x[c].std()
                mask = np.abs((x[c] - m) / (s + 1e-9)) > 3
                x.loc[mask, c] = np.nan
        return x
    
class DataCleaner(BaseEstimator, TransformerMixin):
    """Parse date string, extract day/month/year numric features"""
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        x =  x.copy()
        if 'date_messy' not in x.columns:
            return x
        
        parsed = pd.to_datetime(x['date_messy'], errors='coerce', infer_datetime_format=True)
        x['dt_year'] = parsed.dt.year.fillna(-1)
        x['dt_month'] = parsed.dt.month.fillna(-1)
        x['dt_day'] = parsed.dt.day.fillna(-1)
        #Drop orginal messy dates to avoid issues
        x = x.drop(columns=['date_messy'])
        return x
    

#3. Operation library and parameter grid
OP_NOP = "NOP"
OP_IMPUTE = "IMPUTE"
OP_OUTLIER = "OUTLIER"
OP_SCALE = "SCALE"
OP_ENCODE = "ENCODE"
OP_FEATSEL = "FEATSEL"
OP_DATE = "DATE"

OPS = [OP_NOP, OP_OUTLIER, OP_IMPUTE, OP_SCALE, OP_ENCODE, OP_FEATSEL, OP_DATE]

#Param space for each op - list of options
PARAM_SPACE = {
    OP_NOP: [None],
    OP_OUTLIER: [
        {'method': 'iqr'},
        {'method': 'zscore'}
    ],
    OP_IMPUTE: [
        {'strategy': 'mean'},
        {'strategy': 'median'},
        {'strategy': 'most_frequent'}
    ],
    OP_SCALE: [
        {'method': 'standard'},
        {'method': 'minmax'},
        {'method': 'none'}
    ],
    OP_ENCODE: [
        {'method', 'onehot'},
        {'method', 'ordinal'},
        {'method': 'label_like'} #label_like fallback using ordinal
    ],
    OP_FEATSEL: [
        {'k': 2},
        {'k': 4},
        {'k': 6},
        {'k': None} #None means no selection
    ],
    OP_DATE: [None] #Date cleaner has no params
}

#Helper to get param count for encoding param index
def param_count(op):
    return len(PARAM_SPACE[op])

#4. Decoding chromosome, sklearn pipeline
PIPELINE_LENGTH = 6 #Slots number

def decode_chromosome(chrom):
    """chrom: list of integers length 2 * pipeline_length: [op_idx, param_idx, op_idx, param_idx, ...]
    returns: function apply_pipeline(df) -> transformed_features, or raises"""

    assert len(chrom) == 2 * PIPELINE_LENGTH
    ops_sequence = []
    for i in range(PIPELINE_LENGTH):
        op_idx = chrom[2*i]
        param_idx = chrom[2*i + 1]

        if op_idx < 0 or op_idx >= len(OPS):
            op_idx = 0
        op = OPS[op_idx]

        #Param index
        param_list = PARAM_SPACE[op]
        param_idx = max(0, min(param_idx, len(param_list) - 1))
        params = param_list[param_idx]
        ops_sequence.append((op, params))
    return ops_sequence

def build_transformer_from_ops(ops_seq, df):
    """ops_seq: list of (op, params)
    df: pandas DataFrame (training data) to inspect columns and decide numeric/categorical
    Returns a transformer function that accepts DataFrame and returns numpy X_trans"""

    #Construct a sequence of transformations applied in order using dataframe operations
    def apply_pipeline(df_in):
        x = df_in.copy()

        for op, params in ops_seq:
            if op == OP_NOP:
                continue

            if op == OP_OUTLIER:
                x = OutlierRemover(method=params['method']).transform(x)
            elif op == OP_IMPUTE:
                #Numric + categorical imputer
                num_cols = [c for c in x.columns if x[c].dtype.kind in 'fi']
                cat_cols = [c for c in x.columns if x[c].dtype == object or x[c].dtype.name == 'category']

                if len(num_cols) > 0:
                    impn = SimpleImputer(strategy=params['strategy'])
                    x[num_cols] = impn.fit_transform(x[num_cols])
                
                if len(cat_cols) > 0:
                    #Most frequent works for categorical
                    impc = SimpleImputer(strategy='most_frequent' if params['strategy'] != 'mean' else 'most_frequent')
                    x[cat_cols] = impc.fit_transform(x[num_cols])

            elif op == OP_SCALE:
                method = params['method']
                num_cols = [c for c in x.columns if x[c].dtype.kind in 'fi']

                if method == 'standard' and len(num_cols) > 0:
                    sc = StandardScaler()
                    x[num_cols] = sc.fit_transform(x[num_cols])

                elif method == 'minmax' and len(num_cols) > 0:
                    mm = MinMaxScaler()
                    x[num_cols] = mm.fit_transform(x[num_cols])
                else:
                    pass
            
            elif op == OP_ENCODE:
                method = params['method']
                cat_cols = [c for c in x.columns if x[c].dtype == object or x[c].dtype.name == 'category']

                if len(cat_cols) == 0:
                    continue

                if method == 'onehot':
                    oh = OneHotEncoder(handle_unknown='ignore', sparse=False)
                    arr = oh.fit_transform(c[cat_cols].astype(str))
                    cols = []

                    for i, name in enumerate(cat_cols):
                         #OneHotEncoder doesn't expose names easily here; create synthetic
                         cols.extend([f"{name}_oh_{j}" for j in range(arr.shape[1] // len(cat_cols))])

                    #Fallback: use pandas get_dummies for simplicity
                    dums = pd.get_dummies(x[cat_cols].astype(str), dummy_na=False)
                    x = pd.concat([x.drop(columns=cat_cols), dums.reset_index(drop=True)], axis=1)
                
                elif method == 'ordinal':
                    ord_enc = OrdinalEncoder()
                    x[cat_cols] = ord_enc.fit_transform(x[cat_cols].astype(str))
                
                else:
                    #Lebal like ordinal fallback
                    for c in cat_cols:
                        x[c] = x[c].astype(str).factorize()[0]
            
            elif op == OP_FEATSEL:
                k = params['k']
                if k is None:
                    continue

                #Simple univariate selection using SelectKBest (we need y; we'll attach placeholder and handle later)
                #We'll mark selection to be applied later (because it needs y). For now store selection info
                #We'll simply select top-k numeric features by variance as a proxy (no y dependence) so fitness remains fast
                numric_cols = [c for c in x.columns if x[c].dtype.kind in 'fi']

                if len(numric_cols) <= k:
                    continue
                var_sorted = sorted(numric_cols, key=lambda c: x[c].var() if not x[c].isnull().all() else 0, reverse=True)
                keep = var_sorted[:k]

                #Also keep non-numeric columns (encoded later)
                other_cols = [c for c in x.columns if c not in numric_cols]
                x = x[keep + other_cols]
            
            elif op == OP_DATE:
                x = DataCleaner().transform(x)
            
            else:
                pass
        
        #After all ops, drop any leftover non-numeric columns (if any) by simple encoding
        #Convert object columns to numeric by factorize
        for c in x.columns:
            if x[c].dtype == object:
                x[c] = x[c].astype(str).factorize()[0]
        #Fill any remaining NaNs with 0
        x = x.fillna(0)
        return x
    return apply_pipeline

#5. Genetic algorithm core
POP_SIZE = 30
GENERATIONS = 25
TOURNAMENT_SIZE = 3
CXPB = 0.8
MUTPB = 0.25
ELITISM = 2

def random_chromosome():
    chrom = []
    for _ in range(PIPELINE_LENGTH):
        op_idx = random.randrange(len(OPS))
        op = OPS[op_idx]
        pcount = param_count(op)
        param_idx = random.randrange(pcount)
        chrom.extend([op_idx, param_idx])
    return chrom

def mutate(chrom):
    #Mutate either op_idx or param_idx for a random slot
    new = chrom[:]
    slot = random.randrange(PIPELINE_LENGTH)

    if random.random() < 0.6:
        #Mutate op
        new_op = random.randrange(len(OPS))
        new[2*slot] = new_op

        #Ensure param idx valid
        new[2*slot + 1] = random.randrange(param_count(OPS[new_op]))

    else:
        #Mutate param for same op
        op_idx = new[2*slot]
        new[2*slot + 1] = random.randrange(param_count(ops[op_idx]))
    return new

def crossover(a,b):
    #One-point crossover on gene pairs (slots)
    assert len(a) == len(b)
    point = random.randrange(1, PIPELINE_LENGTH)
    cut = 2*point
    child1 = a[:cut] + b[cut:]
    child2 = b[:cut] + a[cut:]
    return child1, child2

def tournament_select(pop, fitnesses, k= TOURNAMENT_SIZE):
    selected = random.sample(range(len(pop)), k)
    best = max(selected, key=lambda idx: fitnesses[idx])
    return deepcopy(pop[best])

#6. Fitness evaluation
def fitness_of(chrom, df, target_col='target', cv=3, scorer=None):
    ops_seq = decode_chromosome(chrom)
    try:
        build_fn = build_transformer_from_ops(ops_seq, df)
        x = df.drop(columns=[target_col])
        x = df[target_col].values
        x_t = build_fn(x)

        #If too few features -> penalize
        if x_t.shape[1] < 1:
            return 0.0
        
        clf = RandomForestClassifier(n_estimators=50, random_state=random_seed)

        if scorer is None:
            scores = cross_val_score(clf, x_t, y, cv=cv, scoring='accuracy', n_jobs=1)
        else:
            scores = cross_val_score(clf, x_t, y, cv=cv, scoring=scorer, n_jobs=1)
        return float(np.mean(scores))
    
    except Exception as e:
        #Invalid pipeline => very bad fitness
        #Print("fitness error", e)
        return 0.0

#7. GA runner
def run_ga(df, pop_size=POP_SIZE, gens=GENERATIONS):
    #Initialize
    pop = [random_chromosome() for _ in range(pop_size)]
    fitnesses = [fitness_of(ind, df) for ind in pop]
    best_overall = None
    best_score = 1.0
    history = []

    for g in range(gens):
        newpop = []

        #Elitism
        ranked = sorted(list(range(len(pop))), key=lambda i: fitnesses[i], reverse=True)
        elites = [deepcopy(pop[i]) for i in ranked[:ELITISM]]

        #Keep elites into next generation
        while len(newpop) < pop_size - ELITISM:
            #Selection
            parent1 = tournament_select(pop, fitnesses)
            parent2 = tournament_select(pop, fitnesses)

            #Crossover
            if random.random() < CXPB:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = deepcopy(parent1), deepcopy(parent2)
            
            #Mutation
            if random.random() < MUTPB:
                child1 = mutate(child1)
            if random.random() < MUTPB:
                child2 = mutate(child2)
            newpop.append(child1)
            if len(newpop) < pop_size - ELITISM:
                newpop.append(child2)

        #Attach elites
        newpop.extend(elites)
        pop = newpop

        #Evaluate fitness
        fitnesses = [fitness_of(ind, df) for ind in pop]

        #Log best
        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_score = fitnesses[gen_best_idx]
        gen_best = pop[gen_best_idx]
        history.append(gen_best_score)
        
        if gen_best_score > best_score:
            best_score = gen_best_score
            best_overall = deepcopy(gen_best)
        print(f"Gen {g+1}/{gens} — best={gen_best_score:.4f}  overall_best={best_score:.4f}")
    return best_overall, best_score, history

#8. Utilities to pretty-print pipeline
def pretty_pipeline(chrom):
    ops = decode_chromosome(chrom)
    lines = []

    for i, (op,params) in enumerate(ops):
        lines.append(f"{i+1:02d}: {op} {params}")
    return "\n".join(lines)

#9. Main demo routine
if __name__ == "__main__":
    print("Generating synthetic dirty dataset...")
    df = generate_dirty_dataset(n_samples=800)
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    print("\nRunning GA search for pipeline...")
    best_chrom, best_score, history = run_ga(df, pop_size=28, gens=18)

    print("\nBEST SCORE:", best_score)
    print("BEST PIPELINE:")
    print(pretty_pipeline(best_chrom))

    #Show example transformed feature set
    ops_seq = decode_chromosome(best_chrom)
    applier = build_transformer_from_ops(ops_seq, df)
    X_trans = applier(df.drop(columns=['target']))
    print("\nTransformed feature sample shape:", X_trans.shape)
    print(X_trans.head())