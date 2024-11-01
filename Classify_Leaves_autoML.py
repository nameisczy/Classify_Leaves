import os
import pandas as pd
import logging
from autogluon.multimodal import MultiModalPredictor

def fix_root(df, root='kaggle/Classify_Leaves'):
    df['image'] = df['image'].apply(lambda x: os.path.join(root, x))
    return df

def main():
    logging.basicConfig(level=logging.INFO)
    # attach logging to console
    logging.getLogger().addHandler(logging.StreamHandler())

    # load train data
    train_data = fix_root(pd.read_csv('kaggle/Classify_Leaves/train.csv'))

    # Initialize and fit the predictor
    predictor = MultiModalPredictor(label='label', verbosity=3)
    predictor.fit(
        train_data,
        presets='best_quality',
        time_limit=3600*3,
        hyperparameters={'optimization.max_epochs': 30}
    )

    # Load test data
    test = pd.read_csv('kaggle/Classify_Leaves/test.csv')

    # Make predictions on the test data
    pred_test = predictor.predict(fix_root(test.copy()))

    # Save the predictions with image id
    y = pd.concat([test, pred_test.to_frame(name='label')], axis=1)
    y.to_csv('submission_test.csv', index=False)

    # Save the predictor
    predictor.save('predictor.ag')

if __name__ == "__main__":
    main()
