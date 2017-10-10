import numpy as np
import pandas as pd
from pandas import DataFrame, read_csv
from bokeh.models import LinearColorMapper, ColorBar, ColumnDataSource, Select
from bokeh.layouts import widgetbox, row, column, layout
from bokeh.plotting import figure
from bokeh.io import output_notebook, show, output_file, curdoc
from bokeh.plotting import figure
from sklearn import cluster, datasets, mixture
from sklearn.cluster import KMeans,DBSCAN,Birch
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
output_notebook()
digits = pd.read_csv("Wholesale customers data.csv")

attribute_list = [
    'Fresh',
    'Milk',
    'Grocery',  
    'Frozen', 
    'Detergents_Paper', 
    'Delicassen' 
]

algos= [
    'KMeans',
    'DBSCAN',
    'Birch'
]
algo=algos[2]

digits_feature = digits[[#attribute_list[0],
                         attribute_list[1],
                       attribute_list[2],
                        #attribute_list[3],
                         #attribute_list[4],
                        #attribute_list[5]
                       ]]

#########select model#####################
model1 = KMeans(n_clusters=3,random_state=0)
model_fit1 = model1.fit_predict(digits_feature)
model2 = DBSCAN(eps=2000,min_samples=4)
model_fit2 = model2.fit_predict(digits_feature)
model3 = Birch(threshold=0.5, branching_factor=20, n_clusters=2, compute_labels=True)
model_fit3 = model3.fit_predict(digits_feature)


digits['cluster'] = model_fit1
digits.cluster = digits.cluster.astype(str)
#print (digits.head())
#gb_cluster = digits.groupby(['cluster'])
#gb = dict(list(gb_cluster))
#print(gb['0'])
#spectral = np.hstack([Spectral6] * 4)

colors1=[]
for i in model1.labels_:
        if i==-1.0:        #noise
            colors1.append('orange')
        if i==0.0:
            #print ('red')
            colors1.append('red')
        if i==1.0:
            #print ('blue')
            colors1.append('blue')
        if i==2.0:
            colors1.append('green')
        if i==3.0:
            colors1.append('purple')

colors2=[]
for i in model2.labels_:
        if i==-1.0:        #noise
            colors2.append('orange')
        if i==0.0:
            #print ('red')
            colors2.append('red')
        if i==1.0:
            #print ('blue')
            colors2.append('blue')
        if i==2.0:
            colors2.append('green')
        if i==3.0:
            colors2.append('purple')

colors3=[]
for i in model3.labels_:
        if i==-1.0:        #noise
            colors3.append('orange')
        if i==0.0:
            #print ('red')
            colors3.append('red')
        if i==1.0:
            #print ('blue')
            colors3.append('blue')
        if i==2.0:
            colors3.append('green')
        if i==3.0:
            colors3.append('purple')

#colors = [spectral[i] for i in range(0,2)]

p=figure(plot_height=200, plot_width=500, title='KMeans')
actual_data1 = ColumnDataSource(data=dict(x=digits.loc[:,attribute_list[1]],y=digits.loc[:,attribute_list[2]], colors=colors1))
p.circle('x','y',fill_color='colors',line_color='colors',source=actual_data1)
p.xaxis.axis_label = attribute_list[1]
p.yaxis.axis_label = attribute_list[2]


q=figure(plot_height=200, plot_width=500, title='DBScan')
actual_data2 = ColumnDataSource(data=dict(x=digits.loc[:,attribute_list[1]],y=digits.loc[:,attribute_list[2]], colors=colors2))
q.circle('x','y',fill_color='colors',line_color='colors',source=actual_data2)
q.xaxis.axis_label = attribute_list[1]
q.yaxis.axis_label = attribute_list[2]


r=figure(plot_height=200, plot_width=500, title='Birch')
actual_data3 = ColumnDataSource(data=dict(x=digits.loc[:,attribute_list[1]],y=digits.loc[:,attribute_list[2]], colors=colors3))
r.circle('x','y',fill_color='colors',line_color='colors',source=actual_data3)
r.xaxis.axis_label = attribute_list[1]
r.yaxis.axis_label = attribute_list[2]

'''
i=0
for name,group in gb_cluster:
    print (name)
    print (group['cluster'])
    p.circle(x=group['Milk'],y= group['Grocery'], fill_color=colors[i])
    print (i)
    i+=1
    #print (group['Milk'])
#p.circle(x='Milk',y='Grocery',source=gb,fill_color='Navy',alpha=0.4)'''
show(p)
show(q)
show(r)




x_attribute_select = Select(value='Milk',
                          title='Select X attribute:',
                          width=200,
                          options=attribute_list)

y_attribute_select = Select(value='Grocery',
                          title='Select Y attribute:',
                          width=200,
                          options=attribute_list)
show(widgetbox(x_attribute_select))
show(widgetbox(y_attribute_select))
def get_attribute_list(attribute):
    if attribute=='Fresh':
        return digits.loc[:,'Fresh']
    if attribute=='Milk':
        return digits.loc[:,'Milk']
    if attribute=='Grocery':
        return digits.loc[:,'Grocery']
    if attribute=='Frozen':
        return digits.loc[:,'Frozen']
    if attribute=='Detergents_Paper': 
        return digits.loc[:,'Detergents_Paper']
    if attribute=='Delicassen': 
        return digits.loc[:,'Delicassen']        


    
def update_x_attribute(attrname,old,new):
    selected_x_attribute_list=get_attribute_list(x_attribute_select.value)
    selected_y_attribute_list=get_attribute_list(y_attribute_select.value)
    p.xaxis.axis_label = x_attribute_select.value
    q.xaxis.axis_label = x_attribute_select.value   
    r.xaxis.axis_label = x_attribute_select.value   
    actual_data1.data = dict(x=selected_x_attribute_list,y=selected_y_attribute_list,colors=colors1)
    actual_data2.data = dict(x=selected_x_attribute_list,y=selected_y_attribute_list,colors=colors2)
    actual_data3.data = dict(x=selected_x_attribute_list,y=selected_y_attribute_list,colors=colors3)

def update_y_attribute(attrname,old,new):
    selected_x_attribute_list=get_attribute_list(x_attribute_select.value)
    selected_y_attribute_list=get_attribute_list(y_attribute_select.value)
    p.yaxis.axis_label = y_attribute_select.value
    q.yaxis.axis_label = y_attribute_select.value   
    r.yaxis.axis_label = y_attribute_select.value   
    actual_data1.data = dict(x=selected_x_attribute_list,y=selected_y_attribute_list,colors=colors1)
    actual_data2.data = dict(x=selected_x_attribute_list,y=selected_y_attribute_list,colors=colors2)
    actual_data3.data = dict(x=selected_x_attribute_list,y=selected_y_attribute_list,colors=colors3)

x_attribute_select.on_change('value', update_x_attribute)
y_attribute_select.on_change('value', update_y_attribute)


l=layout([
          [row(p,q,r)],
          [row(x_attribute_select,y_attribute_select)]
         ])
curdoc().add_root(l)
