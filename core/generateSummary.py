import pandas as pd


def summarize(results, path):
    predictions_df = pd.DataFrame()
    id2label = {0: 'bar-scale', 1: 'color-stamp', 2: 'detail-labels', 3: 'north-sign'}
    
    for k in results:
        df = pd.DataFrame(results[k]['boxes'].detach().to('cpu').numpy(), columns=['x1', 'y1', 'x2', 'y2'])
        df['labels'] = results[k]['labels'].detach().to('cpu').numpy()
        df['image'] = k.replace('images\\', '')
        # print(df['image'])
        predictions_df = pd.concat([predictions_df, df], ignore_index=True)
    
    image_names = predictions_df['image'].unique()
    labels = [0,1,2,3]

    rows = []
    # Populate the new rows
    for idx, image_name in enumerate(image_names):
        row = {'page_no': idx}
        image_df = predictions_df[predictions_df['image'] == image_name]
        for label in labels:
            label_present = any(image_df['labels'] == label)
            label_count = image_df[image_df['labels'] == label].shape[0]
            row[f'{id2label[label]}'] = label_present
            row[f'{id2label[label]}_count'] = label_count
        rows.append(row)
    # Create a new dataframe from the list of rows
    df2 = pd.DataFrame(rows)
    df2.to_csv(path, index=False)
    