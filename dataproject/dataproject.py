import pandas as pd
import os
from matplotlib import pyplot as plt
from ipywidgets import Output, SelectionSlider, VBox
import ipywidgets as widgets
from IPython.display import display, clear_output

def read_yahoo(input_dir, filename='file.csv'):
    """ Read csv file from Yahoo finance to df\n
    Args: 
        filename (filetype = csv), input_dir = directory of data
    Returns:
        Beautiful financial dataframe
    """
    df = pd.read_csv(os.path.join(input_dir, filename))
    rename_dict = {}
    for i in df.columns:
        rename_dict[i] = i.lower()
        rename_dict[i] = rename_dict[i].replace(' ','_')
    df = df.rename(columns=rename_dict)
    df['daily_return'] = ((df['adj_close'])/(df['adj_close'].shift(1)))-1
    return df

def plot_stock_interactive(merge_final):
    """
    Args: merge_final = cleaned datframe

    Returns interactive plot for cleaned dataframe
    
    """
    out = Output()
    with out:
        def plot_stock_index(stock_index, start_date=min(merge_final.date), end_date=max(merge_final.date)):
            clear_output()
            vertical_lines = merge_final.loc[merge_final.easter_week==1, 'date'].to_numpy()
            y_max = 0.075 # Choose value y axis maximum
            y_min = -0.11 # Choose value y axis minimum
            if start_date <= end_date and stock_index=='omxs': #or stock_index=='nifty': # assert that dates chose make sense
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5)) # initiate plot
                ax.clear()
                I = (merge_final.date>=start_date)&(merge_final.date<=end_date) # dataframe slice condition
                x = merge_final.loc[I, 'date'] # x values
                y = merge_final.loc[I, f'daily_return_{stock_index}'] # y-values
                ax.set_ylim(y_min, y_max) # set y axis limits
                ax.plot(x, y, linewidth=1) # plot
                ax.set_xlim(start_date, end_date) # set x axis limits
                ax.set_xticks(x[::len(merge_final.loc[I,:])//7]) #set x ticks to vary with chose periode
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                ax.grid(True, alpha=0.3)
                ax.set_title(f'Daily returns for {stock_index}')
                ax.vlines(vertical_lines, y_min, y_max, color='#39ff14',
                           alpha=0.3,
                           linewidth=6, label="Days during easter")
                ax.legend(loc="lower center")
                plt.show()
            elif stock_index == 'both' and start_date <= end_date:
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
                ax.clear()
                I = (merge_final.date>=start_date)&(merge_final.date<=end_date)
                for i in ['omxs', 'nifty']:
                    x = merge_final.loc[I, 'date']
                    y = merge_final.loc[I, f'daily_return_{i}']
                    ax.set_ylim(y_min, y_max)
                    ax.plot(x, y, linewidth=1)
                    clear_output()
                ax.set_xlim(start_date, end_date)
                ax.set_xticks(x[::len(merge_final.loc[I,:])//7])
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                ax.grid(True, alpha=0.3)
                ax.vlines(vertical_lines, y_min, y_max,
                           color='#39ff14',
                           alpha=0.3, linewidth=6, label="Days during easter")
                ax.set_title(f"Daily returns for 'OMXS', 'NIFTY FIFTY'")
                ax.legend(loc="lower center")
            else: 
                 print("Hello mate, please choose a valid range of dates")
        
    out = widgets.interact(plot_stock_index, stock_index=['omxs', "nifty"], #start_date=merge_final.date, end_date=merge_final.date)#,
                              start_date = SelectionSlider(value= pd.to_datetime("2016-05-03 00:00:00"), options=merge_final.date, step=1),
                                end_date = SelectionSlider(value=pd.to_datetime("2016-06-09 00:00:00"), options=merge_final.date, step=1));
    
    return  display(out)


#from ipywidgets import Output, IntSlider, VBox
#from IPython.display import clear_output
#out = Output()
#
#slider = IntSlider()
#
#def square(change):
#    with out:
#        clear_output()
#        print(change.new*change.new)
#
#slider.observe(square, 'value')
#slider.value = 50
#display(VBox([slider, out]))